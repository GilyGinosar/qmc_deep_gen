# train_qmc_all_unified.py
import os, glob, h5py
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==== project imports ====
from data.bird_data import bird_data
from models.sampling import gen_fib_basis, gen_korobov_basis         # latent grid for 2D
from models.utils import get_decoder_arch
from models.qmc_base import QMCLVM
from train.losses import binary_evidence, binary_lp
from train.train import train_loop, test_epoch

# =========================
# Config (edit to taste)
# =========================
ROOTS_PER_FAMILY = {
    1: [r"D:\Data\Data_vae\235", r"D:\Data\Data_vae\237"],
    #2: [r"D:\Data\113", r"D:\Data\114", r"D:\Data\115", r"D:\Data\116"],
}
FAMILIES       = [1]      # which families to include
SPECS_PER_FILE = 100
TEST_SIZE      = 0.20
SPLIT_SEED     = 92
N_WORKERS      = 0           # Win-safe; raise if Linux
OUT_DIR_ROOT   = r"D:\data\model_checkpoints"

# Toggle: conditional vs unconditional
COND           = False        # <--- ################### flip this
CF             = "rule3_bands"  # only used if COND=True

LATENT_DIM     = 3
# Fibonacci 2D
M_FIB          = 15          # latent grid density (Fibonacci)
# Korobov 3D
N_LATENT_POINTS = 1021  # or 2039, 4093
a_korobov = 76

N_EPOCHS       = 10          # training epochs

# Batch size
BATCH          = 1 if COND else 64


# =========================
# Utilities
# =========================
def spec_to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).to(torch.float32).unsqueeze(0)


def load_gerbils_multi(gerbil_filepath, specs_per_file, families,
                       test_size=0.2, seed=92, check=True):
    """
    takes data from multiple experiment folders
    Returns: (train_fns, test_fns), (train_ids, test_ids), specs_per_file
    """
    try:
        len(families)
    except Exception:
        families = [families]

    if isinstance(gerbil_filepath, dict):
        def get_roots(fam): return gerbil_filepath.get(fam, [])
    elif isinstance(gerbil_filepath, (list, tuple)):
        def get_roots(fam): return list(gerbil_filepath)
    else:
        def get_roots(fam): return [gerbil_filepath]

    specs_in_file = []
    all_family_specs = []
    all_family_ids = []

    for ii, family in enumerate(families):
        print(f"[load] family{family}")
        roots = get_roots(family)

        fam_spec_fns = []
        for root in roots:
            spec_dir = os.path.join(root, 'processed-data', f"family{family}")
            spec_fns = glob.glob(os.path.join(spec_dir, '*.hdf5'))
            fam_spec_fns.extend(spec_fns)

            if check:
                for spec_fn in tqdm(spec_fns, total=len(spec_fns), desc=f"checking {spec_dir}"):
                    with h5py.File(spec_fn, 'r') as f:
                        sif = len(f['specs'])
                        specs_in_file.append(sif)

        all_family_specs += fam_spec_fns
        all_family_ids.append(ii * np.ones((len(fam_spec_fns),)))

    if check and specs_in_file:
        num_specs = np.unique(specs_in_file)
        assert len(num_specs) == 1, f"Files have different numbers of specs: {num_specs}"
        if num_specs[0] != specs_per_file:
            print(f"[warn] expected {specs_per_file}, found {num_specs[0]} → updating")
            specs_per_file = int(num_specs[0])

    all_family_ids = np.hstack(all_family_ids) if len(all_family_ids) else np.array([])

    if test_size > 0 and len(all_family_specs) > 0:
        tr_fns, te_fns, tr_ids, te_ids = train_test_split(
            all_family_specs, all_family_ids, test_size=test_size, random_state=seed
        )
    else:
        tr_fns, te_fns = all_family_specs, all_family_specs
        tr_ids, te_ids = all_family_ids, all_family_ids

    return (tr_fns, te_fns), (tr_ids, te_ids), specs_per_file


def build_datasets_and_loaders(roots_per_family, families, specs_per_file,
                               batch_size, n_workers, cond: bool, cf: str):
    (train_fns, test_fns), (train_ids, test_ids), specs_per_file = load_gerbils_multi(
        gerbil_filepath=roots_per_family,
        specs_per_file=specs_per_file,
        families=families,
        test_size=TEST_SIZE,
        seed=SPLIT_SEED,
        check=True,
    )

    # Datasets
    if cond:
        train_ds = bird_data(train_fns, train_ids, specs_per_file=specs_per_file,
                             transform=spec_to_tensor, conditional=True, conditional_factor=cf)
        test_ds  = bird_data(test_fns,  test_ids,  specs_per_file=specs_per_file,
                             transform=spec_to_tensor, conditional=True, conditional_factor=cf)
    else:
        train_ds = bird_data(train_fns, train_ids, specs_per_file=specs_per_file,
                             transform=spec_to_tensor, conditional=False)
        test_ds  = bird_data(test_fns,  test_ids,  specs_per_file=specs_per_file,
                             transform=spec_to_tensor, conditional=False)

    use_cuda = torch.cuda.is_available()
    pin = bool(use_cuda)

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=n_workers, pin_memory=pin)

    return train_loader, test_loader, specs_per_file


def make_decoder(dataset_name: str, latent_dim: int, cond: bool, cond_dim: int):
    if cond:
        return get_decoder_arch(dataset_name=dataset_name,
                                latent_dim=latent_dim,
                                arch="conditional_qmc",
                                cond_dim=cond_dim)
    # Unconditional: try plain 'qmc'; if not present, fall back to 'conditional_qmc' with cond_dim=0
    try:
        return get_decoder_arch(dataset_name=dataset_name,
                                latent_dim=latent_dim,
                                arch="qmc")
    except Exception:
        return get_decoder_arch(dataset_name=dataset_name,
                                latent_dim=latent_dim,
                                arch="conditional_qmc",
                                cond_dim=0)


# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR_ROOT, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")

    # 1) build data loaders
    train_loader, test_loader, specs_per_file = build_datasets_and_loaders(
        ROOTS_PER_FAMILY, FAMILIES, SPECS_PER_FILE, BATCH, N_WORKERS, COND, CF
    )

    # 2) build model (make_decoder)
    cond_dim = (3 if COND else 0)
    decoder = make_decoder(dataset_name="gerbil_ava", latent_dim=LATENT_DIM,
                           cond=COND, cond_dim=cond_dim)
    model = QMCLVM(latent_dim=LATENT_DIM, device=device, decoder=decoder)

    # sanity check - print first Linear layer
    # first_linear = None
    # for m in model.decoder.modules():
    #     if isinstance(m, nn.Linear):
    #         first_linear = m
    #         break
    # if first_linear is not None:
    #     print("[decoder[0]]", first_linear)

    # print decoder
    print("Decoder architecture, top-level layers:")
    for i, layer in enumerate(model.decoder):
        print(f"[{i}] {layer}")

    # 3) training setup - latent grid (2D: Fiboncci; 3D: Korobov) + loss function
    latent_grid = gen_korobov_basis(a=a_korobov,num_dims=LATENT_DIM,num_points=N_LATENT_POINTS).to(device).float()
    # latent_grid = gen_fib_basis(m=M_FIB)  # 2D grid
    qmc_loss_func = binary_evidence
    qmc_lp        = binary_lp

    # 4) out dir
    tag = "cond" if COND else "uncond"
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_DIR_ROOT, f"{tag}_all_{run_stamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")

    # 5) train
    model, opt, losses = train_loop(
        model,
        train_loader,
        latent_grid.to(device),
        qmc_loss_func,
        nEpochs=N_EPOCHS,
        verbose=True,
        conditional=COND,     # <--- key toggle
        out_dir=out_dir
    )

    # 6) eval on test split (same loss)
    with torch.no_grad():
        _ = test_epoch(
            model,
            test_loader,
            latent_grid.to(device),
            qmc_loss_func,
            conditional=COND
        )

    print("[done]")


if __name__ == "__main__":
    # Windows-safe entrypoint
    import torch.multiprocessing as mp
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
