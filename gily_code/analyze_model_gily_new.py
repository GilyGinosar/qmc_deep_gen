# eval_qmc_uncond_all.py
import os, glob, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.serialization import add_safe_globals
from torch.torch_version import TorchVersion
add_safe_globals([TorchVersion])  # allowlist for safe unpickling

# ==== project imports ====
from data.bird_data import bird_data, calc_mean_freq
from models.sampling import gen_fib_basis
from models.utils import get_decoder_arch
from models.qmc_base import QMCLVM
from train.losses import binary_evidence, binary_lp
from train.train import test_epoch
from plotting.visualize import qmc_train_plot, format_plot_axis,model_grid_plot

# --------- config ----------
DATA_ROOT = {
    1: [r"D:\Data\235", r"D:\Data\237"],
    2: [r"D:\Data\113", r"D:\Data\114", r"D:\Data\115", r"D:\Data\116"],
}
FAMILIES        = [1, 2]                 # which families to include
CKPT_DIR        = r"D:\data\Figs_uncond_all\model_checkpoints\uncond_all_20251015_130934"  # <-- single model folder
OUT_FIG_DIR     = r"D:\Data\Figs_uncond_all"


import torch, glob, os

folder = CKPT_DIR
ckpts = sorted(glob.glob(os.path.join(folder, "final_*.pt")) +
               glob.glob(os.path.join(folder, "ckpt_*.pt")), key=os.path.getmtime)
assert ckpts, f"No ckpts in {folder}"
path = ckpts[-1]
sd = torch.load(path, map_location="cpu", weights_only=True)
if "model_state_dict" in sd: sd = sd["model_state_dict"]

w = sd.get("decoder.0.weight", None)
print(path, "decoder.0.weight shape =", None if w is None else tuple(w.shape))




BATCH_SIZE      = 64
NUM_WORKERS     = 0
LATENT_DIM      = 2
M_FIB           = 15
SPECS_PER_FILE  = 100
TEST_SIZE       = 0.20
SPLIT_SEED      = 92

# spectrogram axis (used only if you later want mean-freq maps)
MIN_FREQ_HZ     = 500
MAX_FREQ_HZ     = 62_500
NUM_FREQ_BINS   = 128
FREQ_AXIS_GLOBAL = np.linspace(MIN_FREQ_HZ, MAX_FREQ_HZ, NUM_FREQ_BINS, dtype=np.float64)

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# ---------- helpers ----------
def spec_to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).to(torch.float32).unsqueeze(0)

def latest_checkpoint(ckpt_dir: str) -> str:
    finals = sorted(glob.glob(os.path.join(ckpt_dir, "final_*.pt")), key=os.path.getmtime)
    if finals: return finals[-1]
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")), key=os.path.getmtime)
    if ckpts: return ckpts[-1]
    raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

def strip_dataparallel_prefix(sd):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

def load_gerbils_multi(gerbil_filepath, specs_per_file, families=[2],
                       test_size=0.2, seed=92, check=True):
    import h5py
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    try: len(families)
    except Exception: families = [families]

    if isinstance(gerbil_filepath, dict):
        def get_roots(fam): return gerbil_filepath.get(fam, [])
    elif isinstance(gerbil_filepath, (list, tuple)):
        def get_roots(fam): return list(gerbil_filepath)
    else:
        def get_roots(fam): return [gerbil_filepath]

    specs_in_file, all_family_specs, all_family_ids = [], [], []
    for ii, family in enumerate(families):
        print(f"loading family{family}")
        roots = get_roots(family)
        fam_spec_fns = []
        for root in roots:
            spec_dir = os.path.join(root, 'processed-data', f"family{family}")
            spec_fns = glob.glob(os.path.join(spec_dir, '*.hdf5'))
            fam_spec_fns.extend(spec_fns)
            if check:
                for spec_fn in tqdm(spec_fns, total=len(spec_fns), desc=f"checking {spec_dir}"):
                    with h5py.File(spec_fn, 'r') as f:
                        specs_in_file.append(len(f['specs']))
        all_family_specs += fam_spec_fns
        all_family_ids.append(ii * np.ones((len(fam_spec_fns),)))

    if check and specs_in_file:
        num_specs = np.unique(specs_in_file)
        assert len(num_specs) == 1, f"Different #specs per file: {num_specs}"
        if num_specs[0] != specs_per_file:
            print(f"[warn] expected {specs_per_file}, found {num_specs[0]}; updating")
            specs_per_file = int(num_specs[0])

    all_family_ids = np.hstack(all_family_ids) if len(all_family_ids) else np.array([])

    if test_size > 0 and len(all_family_specs) > 0:
        train_fns, test_fns, train_ids, test_ids = train_test_split(
            all_family_specs, all_family_ids, test_size=test_size, random_state=seed
        )
    else:
        train_fns, test_fns = all_family_specs, all_family_specs
        train_ids, test_ids = all_family_ids, all_family_ids

    return (train_fns, test_fns), (train_ids, test_ids), specs_per_file

def build_plain_loaders():
    """Single unconditional dataset/loaders (no bins)."""
    (train_fns, test_fns), (train_ids, test_ids), specs_per_file = load_gerbils_multi(
        gerbil_filepath=DATA_ROOT,
        specs_per_file=SPECS_PER_FILE,
        families=FAMILIES,
        test_size=TEST_SIZE,
        seed=SPLIT_SEED,
        check=True,
    )
    train_ds = bird_data(train_fns, train_ids, specs_per_file=specs_per_file,
                         transform=spec_to_tensor, conditional=False)
    test_ds  = bird_data(test_fns,  test_ids,  specs_per_file=specs_per_file,
                         transform=spec_to_tensor, conditional=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    return train_loader, test_loader, train_fns, test_fns

def _read_locations_for_file(h5_path):
    import h5py
    with h5py.File(h5_path, 'r') as f:
        locs = f['locations'][:]
    return np.array([x.decode('ASCII') for x in locs], dtype=object)

def build_locations_vector(file_list):
    out = []
    for p in file_list:
        out.extend(_read_locations_for_file(p))
    return np.array(out, dtype=object)

def _loc_bucketize(loc_str_array: np.ndarray) -> np.ndarray:
    s = np.char.lower(loc_str_array.astype(str))
    arena_mask = np.logical_or(np.char.find(s, "arena_1") >= 0,
                               np.char.find(s, "arena_2") >= 0)
    return np.where(arena_mask, "arena", "underground")

def _nice_colors(n):
    base = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
    if n <= len(base): return base[:n]
    return [base[i % len(base)] for i in range(n)]

def plot_by_family(emb_xy: np.ndarray, fam_labels: np.ndarray, split_name: str, out_dir: str):
    fams = np.unique(fam_labels)
    colors = _nice_colors(len(fams))
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    for col, fam in zip(colors, fams):
        m = (fam_labels == fam)
        ax.scatter(emb_xy[m, 0], emb_xy[m, 1], s=6, marker=".", alpha=0.85, c=col, label=f"family {int(fam)}")
    format_plot_axis(ax, xlim=(0,1), ylim=(0,1),
                     xlabel="Latent dim 1", ylabel="Latent dim 2",
                     title=f"(Unconditional) Embeddings by family — {split_name}")
    ax.legend(frameon=False, loc="best", markerscale=1.5)
    plt.tight_layout()
    fn = os.path.join(out_dir, f"embeddings_uncond_by_family_{split_name}.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_by_location(emb_xy: np.ndarray, loc_bucket: np.ndarray, split_name: str, out_dir: str):
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    for name, col in [("arena", "C0"), ("underground", "C3")]:
        m = (loc_bucket == name)
        ax.scatter(emb_xy[m, 0], emb_xy[m, 1], s=6, marker=".", alpha=0.85, c=col, label=name)
    format_plot_axis(ax, xlim=(0,1), ylim=(0,1),
                     xlabel="Latent dim 1", ylabel="Latent dim 2",
                     title=f"(Unconditional) Embeddings by location — {split_name}")
    ax.legend(frameon=False, loc="best", markerscale=1.5)
    plt.tight_layout()
    fn = os.path.join(out_dir, f"embeddings_uncond_by_location_{split_name}.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close(fig)

def rebuild_uncond_model(device):
    decoder = get_decoder_arch(dataset_name="gerbil_ava", latent_dim=LATENT_DIM, arch="qmc")
    return QMCLVM(latent_dim=LATENT_DIM, device=device, decoder=decoder)

def load_weights(model, ckpt_path, device):
    # Prefer safe load; fall back only if you trust the file.
    try:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(blob, dict) and "model_state_dict" in blob:
            state = blob["model_state_dict"]; train_losses = blob.get("losses", []); epoch = blob.get("epoch", None)
        else:
            state = blob; train_losses = []; epoch = None
    except Exception as e:
        print(f"[load] weights_only=True failed ({e}). Falling back to weights_only=False (trusted file).")
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and "model_state_dict" in blob:
            state = blob["model_state_dict"]; train_losses = blob.get("losses", []); epoch = blob.get("epoch", None)
        else:
            state = blob; train_losses = []; epoch = None

    state = strip_dataparallel_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] missing: {missing}\n[load] unexpected: {unexpected}")
    model.to(device); model.eval()
    return train_losses, epoch

def main():
    torch.manual_seed(92); np.random.seed(92)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(92)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 1) Data (unconditional, full)
    train_loader, test_loader, train_fns, test_fns = build_plain_loaders()

    # 2) Model + checkpoint
    ckpt_path = latest_checkpoint(CKPT_DIR)
    print("loading ckpt:", ckpt_path)
    model = rebuild_uncond_model(device)
    train_losses, epoch = load_weights(model, ckpt_path, device)

    # --- latent grid visualization (unconditional model) ---
    with torch.no_grad():
        model_grid_plot(
            model,
            n_samples_dim=20,
            origin="lower",
            cm="inferno",
            show=False,
            fn=os.path.join(OUT_FIG_DIR, "decoder_grid_uncond.png")
        )

    # # 3) Eval loss
    # latent_grid = gen_fib_basis(m=M_FIB)
    # with torch.no_grad():
    #     test_losses = test_epoch(model, test_loader, latent_grid.to(device), binary_evidence, conditional=False)
    # os.makedirs(OUT_FIG_DIR, exist_ok=True)
    # qmc_train_plot(train_losses, test_losses,
    #                save_fn=os.path.join(OUT_FIG_DIR, "train_vs_test_uncond.png"),
    #                show=False)
    #
    # # 4) Embeddings (unconditional; no c)
    # with torch.no_grad():
    #     emb_te, lab_te = model.embed_data(latent_grid.to(device), test_loader, binary_lp,
    #                                       embed_type="rqmc", n_samples=5)
    #     emb_tr, lab_tr = model.embed_data(latent_grid.to(device), train_loader, binary_lp,
    #                                       embed_type="rqmc", n_samples=5)
    #
    # # 5) Location vectors aligned to dataset order (file-major, no shuffle)
    # train_locs_all = build_locations_vector(train_fns)
    # test_locs_all  = build_locations_vector(test_fns)
    # loc_tr_bucket  = _loc_bucketize(train_locs_all)
    # loc_te_bucket  = _loc_bucketize(test_locs_all)
    #
    # # 6) Plots — families & locations (train & test)
    # plot_by_family(emb_tr, lab_tr, split_name="train", out_dir=OUT_FIG_DIR)
    # plot_by_family(emb_te, lab_te, split_name="test",  out_dir=OUT_FIG_DIR)
    #
    # plot_by_location(emb_tr, loc_tr_bucket, split_name="train", out_dir=OUT_FIG_DIR)
    # plot_by_location(emb_te, loc_te_bucket, split_name="test",  out_dir=OUT_FIG_DIR)
    #
    # # Meta
    # with open(os.path.join(OUT_FIG_DIR, "eval_meta_uncond.json"), "w") as f:
    #     json.dump({"ckpt": os.path.abspath(ckpt_path),
    #                "epoch": epoch,
    #                "n_test_batches": len(test_loader),
    #                "device": str(device),
    #                "time": time.strftime("%Y-%m-%d %H:%M:%S")},
    #               f, indent=2)
    # print(f"Saved figures → {OUT_FIG_DIR}")

if __name__ == "__main__":
    import time
    import torch.multiprocessing as mp
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
