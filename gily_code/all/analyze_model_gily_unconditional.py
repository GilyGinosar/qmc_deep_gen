# eval_qmc_uncond_bins.py
import os, glob, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
from plotting.visualize import model_grid_plot, qmc_train_plot, format_plot_axis

# --------- config ----------
DATA_ROOT = {
    1: [r"D:\Data\237"],    # same split you used for training
}
FAMILIES        = [1]
CKPT_ROOT       = r"D:\data\model_checkpoints"   # parent that contains bin0/, bin1/, bin2/
OUT_FIG_DIR     = r"D:\Data\Figs_uncond_eval"

BATCH_SIZE      = 64
NUM_WORKERS     = 0
LATENT_DIM      = 2
M_FIB           = 15
SPECS_PER_FILE  = 100
TEST_SIZE       = 0.20
SPLIT_SEED      = 92

# spectrogram axis (used for mean-frequency visual)
MIN_FREQ_HZ     = 500
MAX_FREQ_HZ     = 62_500
NUM_FREQ_BINS   = 128
FREQ_AXIS_GLOBAL = np.linspace(MIN_FREQ_HZ, MAX_FREQ_HZ, NUM_FREQ_BINS, dtype=np.float64)

os.makedirs(OUT_FIG_DIR, exist_ok=True)


# ---------- helpers ----------
def spec_to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).to(torch.float32).unsqueeze(0)

def latest_checkpoint_in(folder: str) -> str:
    finals = sorted(glob.glob(os.path.join(folder, "final_*.pt")), key=os.path.getmtime)
    if finals: return finals[-1]
    ckpts = sorted(glob.glob(os.path.join(folder, "ckpt_*.pt")), key=os.path.getmtime)
    if ckpts: return ckpts[-1]
    raise FileNotFoundError(f"No checkpoints in {folder}")

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


def build_per_bin_plain_loaders():
    """Use rule3_bands once to bin items; then return PLAIN (unconditional) loaders per bin."""
    # 1) split filenames same as training
    (train_fns, test_fns), (train_ids, test_ids), specs_per_file = load_gerbils_multi(
        gerbil_filepath=DATA_ROOT,
        specs_per_file=SPECS_PER_FILE,
        families=FAMILIES,
        test_size=TEST_SIZE,
        seed=SPLIT_SEED,
        check=True,
    )

    # 2) conditional pass to get bin index
    ds_tr_cond = bird_data(train_fns, train_ids, specs_per_file=specs_per_file,
                           transform=spec_to_tensor, conditional=True, conditional_factor="rule3_bands")
    ds_te_cond = bird_data(test_fns,  test_ids,  specs_per_file=specs_per_file,
                           transform=spec_to_tensor, conditional=True, conditional_factor="rule3_bands")

    # 3) plain datasets for actual evaluation (unconditional)
    ds_tr_plain = bird_data(train_fns, train_ids, specs_per_file=specs_per_file,
                            transform=spec_to_tensor, conditional=False)
    ds_te_plain = bird_data(test_fns,  test_ids,  specs_per_file=specs_per_file,
                            transform=spec_to_tensor, conditional=False)

    # 4) collect indices per bin
    bin_to_idx_tr = {0: [], 1: [], 2: []}
    for i in range(len(ds_tr_cond)):
        _, c, _ = ds_tr_cond[i]
        bin_to_idx_tr[int(c.argmax().item())].append(i)

    bin_to_idx_te = {0: [], 1: [], 2: []}
    for i in range(len(ds_te_cond)):
        _, c, _ = ds_te_cond[i]
        bin_to_idx_te[int(c.argmax().item())].append(i)

    # 5) build per-bin plain loaders
    use_cuda = torch.cuda.is_available()
    pin = bool(use_cuda)

    loaders = {}
    for b in (0, 1, 2):
        tr_subset = Subset(ds_tr_plain, bin_to_idx_tr[b])
        te_subset = Subset(ds_te_plain, bin_to_idx_te[b])
        ntr, nte = len(tr_subset), len(te_subset)
        if ntr == 0:
            print(f"BIN {b}: empty (train=0, test={nte}) — skipping.")
            loaders[b] = {"train": None, "test": None, "n_train": 0, "n_test": nte}
            continue
        bs_tr = min(BATCH_SIZE, ntr)
        bs_te = min(BATCH_SIZE, max(1, nte))
        loaders[b] = dict(
            train=DataLoader(tr_subset, batch_size=bs_tr, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin),
            test=DataLoader(te_subset, batch_size=bs_te, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin),
            n_train=ntr, n_test=nte
        )
    return loaders


def rebuild_uncond_model(device):
    decoder = get_decoder_arch(dataset_name="gerbil_ava", latent_dim=LATENT_DIM, arch="qmc")  # UNCONDITIONAL
    model = QMCLVM(latent_dim=LATENT_DIM, device=device, decoder=decoder)
    return model


def load_weights(model, ckpt_path, device):
    """
    Loads checkpoints saved either as a full dict (with 'model_state_dict', 'losses', 'epoch', ...)
    or as a raw state_dict. Works with PyTorch 2.6 weights_only behavior.
    """
    # 1) Try safe load (weights_only=True). This is the new default; we pass it explicitly.
    try:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(blob, dict) and "model_state_dict" in blob:
            state = blob["model_state_dict"]
            train_losses = blob.get("losses", [])
            epoch = blob.get("epoch", None)
        else:
            # weights_only likely returned the raw state_dict
            state = blob
            train_losses = []
            epoch = None
    except Exception as e:
        print(f"[load] weights_only=True failed ({e}). Falling back to weights_only=False (trusted file).")
        # 2) Fallback to full pickle (ONLY if you trust the file)
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and "model_state_dict" in blob:
            state = blob["model_state_dict"]
            train_losses = blob.get("losses", [])
            epoch = blob.get("epoch", None)
        else:
            # Extremely rare, but handle it
            state = blob
            train_losses = []
            epoch = None

    # Strip possible DataParallel prefix and load
    state = strip_dataparallel_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] missing: {missing}\n[load] unexpected: {unexpected}")

    model.to(device)
    model.eval()
    return train_losses, epoch



def meanfreq_map_over_grid(model, z_grid):
    """Decode grid → mean frequency (Hz) per sample."""
    with torch.no_grad():
        X = model(z_grid, random=False, mod=False)  # [K,1,F,T]
    vals = []
    Fx = FREQ_AXIS_GLOBAL
    for i in range(X.shape[0]):
        spec = X[i, 0].detach().cpu().numpy()
        vals.append(calc_mean_freq(spec, freq_axis=Fx))
    return np.asarray(vals, dtype=np.float64)


def plot_meanfreq_scatter(Z01, vals_hz, title, out_png):
    vals_khz = vals_hz / 1000.0
    vmin_khz, vmax_khz = float(np.min(vals_khz)), float(np.max(vals_khz))
    norm = colors.Normalize(vmin=vmin_khz, vmax=vmax_khz)

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 5.2))
    sc = ax.scatter(Z01[:, 0], Z01[:, 1], c=vals_khz, s=120, alpha=0.98,
                    cmap="viridis", norm=norm, edgecolors="none", linewidths=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal", "box")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label("Mean frequency (kHz)")
    fig.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bin_train_test(emb_train, emb_test, bin_id, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    ax = axes[0]
    ax.scatter(emb_test[:, 0], emb_test[:, 1], s=3, marker=".", alpha=0.7, c="C0")
    format_plot_axis(ax, xlim=(0, 1), ylim=(0, 1),
                     xlabel="Latent dim 1", ylabel="Latent dim 2",
                     title=f"Bin {bin_id} — Test")

    ax = axes[1]
    ax.scatter(emb_train[:, 0], emb_train[:, 1], s=3, marker=".", alpha=0.7, c="C0")
    format_plot_axis(ax, xlim=(0, 1), ylim=(0, 1),
                     xlabel="Latent dim 1", ylabel="Latent dim 2",
                     title=f"Bin {bin_id} — Train")

    plt.tight_layout()
    fn = os.path.join(out_dir, f"embeddings_train_test_bin{bin_id}.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    # reproducibility (optional)
    torch.manual_seed(92); np.random.seed(92)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(92)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Build per-bin plain loaders (unconditional)
    loaders = build_per_bin_plain_loaders()

    # latent grid & losses (same as training)
    latent_grid = gen_fib_basis(m=M_FIB)      # [K,2] in [0,1) space (use %1 if you visualize)
    qmc_loss_func = binary_evidence
    qmc_lp        = binary_lp

    # For each bin: load model, eval, figures
    for b in (0, 1, 2):
        if loaders[b]["n_train"] == 0:
            print(f"\n=== BIN {b} skipped (no training items) ===")
            continue

        print(f"\n=== Evaluating BIN {b} model ===")
        bin_dir = os.path.join(CKPT_ROOT, f"bin{b}")
        ckpt_path = latest_checkpoint_in(bin_dir)
        print("loading:", ckpt_path)

        model = rebuild_uncond_model(device)
        train_losses, epoch = load_weights(model, ckpt_path, device)

        # 1) test loss on that bin’s test set
        with torch.no_grad():
            test_losses = test_epoch(
                model,
                loaders[b]["test"],
                latent_grid.to(device),
                qmc_loss_func,
                conditional=False,  # UNCONDITIONAL
            )
        # save small meta + losses
        out_dir = os.path.join(bin_dir, f"eval_epoch{epoch if epoch is not None else 'NA'}")
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "test_losses.npy"), np.asarray(test_losses, np.float32))
        with open(os.path.join(out_dir, "eval_meta.json"), "w") as f:
            json.dump(
                {"ckpt": os.path.abspath(ckpt_path),
                 "epoch": epoch,
                 "n_test_batches": len(loaders[b]["test"]),
                 "device": str(device),
                 "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                f, indent=2,
            )

        # 2) training vs test evidence plot
        qmc_train_plot(train_losses, test_losses,
                       save_fn=os.path.join(out_dir, f"train_vs_test_bin{b}.png"),
                       show=False)

        # 3) decoder grid visualization (per-bin model)
        with torch.no_grad():
            model_grid_plot(model, n_samples_dim=20, origin="lower", cm="inferno",
                            show=False, fn=os.path.join(out_dir, f"decoder_grid_bin{b}.png"))

        # 4) embeddings: TEST & TRAIN for this bin (no c)
        with torch.no_grad():
            emb_te, lab_te = model.embed_data(
                latent_grid.to(device),
                loaders[b]["test"],
                qmc_lp,
                embed_type="rqmc",
                n_samples=5,
            )
            emb_tr, lab_tr = model.embed_data(
                latent_grid.to(device),
                loaders[b]["train"],
                qmc_lp,
                embed_type="rqmc",
                n_samples=5,
            )
        plot_bin_train_test(emb_tr, emb_te, b, out_dir)

        # 5) mean-frequency map over latent grid (per-bin model)
        with torch.no_grad():
            z_grid = gen_fib_basis(m=15).to(device)
            Z01 = (z_grid % 1).detach().cpu().numpy()
            mf_hz = meanfreq_map_over_grid(model, z_grid)
        plot_meanfreq_scatter(Z01, mf_hz, title=f"Bin {b} — mean freq (kHz)",
                              out_png=os.path.join(out_dir, f"meanfreq_grid_bin{b}.png"))

    print("\nAll done. Figures saved under each bin’s eval folder in:", CKPT_ROOT)


if __name__ == "__main__":
    import time
    import torch.multiprocessing as mp
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
