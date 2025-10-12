# eval_qmc.py
import os, glob, time, json
import numpy as np
import torch
from sympy.polys.matrices.dense import ddm_irref_den
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn as nn
from scipy.ndimage import zoom

from torch.utils.data import Dataset, Subset

# ==== project imports (same stack you used for training) ====
from data.bird_data import load_gerbils, bird_data
from models.sampling import gen_fib_basis  # latent grid for 2D
from models.utils import get_decoder_arch
from models.qmc_base import QMCLVM
from train.losses import binary_evidence, binary_lp
from train.train import test_epoch

# (optional) plotting helpers
from plotting.visualize import model_grid_plot, qmc_train_plot, format_plot_axis
import matplotlib.pyplot as plt
from torch.serialization import add_safe_globals  # or use the context manager shown below
from torch.torch_version import TorchVersion

import h5py
from tqdm import tqdm
from sklearn.model_selection import train_test_split

add_safe_globals([TorchVersion])  # allowlist just this class


# --------- config ----------
DATA_ROOT = {
    1: [r"D:\Data\235", r"D:\Data\237"],
    2: [r"D:\Data\113", r"D:\Data\114", r"D:\Data\115", r"D:\Data\116"],
}
TEST_FAMILY_IDS = [1,2]
CKPT_DIR        = fr"D:\Data\model_checkpoints"    # where train_loop saved checkpoints
BATCH_SIZE      = 64
NUM_WORKERS     = 2                                # Windows-safe
LATENT_DIM      = 2
M_FIB           = 15                               # same latent grid density
SPECS_PER_FILE  = 100
TEST_SIZE       = 0.20
SPLIT_SEED      = 92

# --- conditional binning params (match training) ---
MIN_FREQ_HZ    = 500
MAX_FREQ_HZ    = 62500
NUM_FREQ_BINS  = 128
COND_FACTOR    = "freq_bin1h"   # the one-hot(3) we added in bird_data
COND_DIM       = 3                   # one-hot length
FREQ_AXIS_GLOBAL = np.linspace(MIN_FREQ_HZ, MAX_FREQ_HZ, NUM_FREQ_BINS, dtype=np.float64)


out_dir_fig = r"D:\Data\Figs"
os.makedirs(out_dir_fig, exist_ok=True)


# -----------------------------------------------

# takes a dataset that yields (spec, c, label) and exposes (spec, label) so embed_data can stay unchanged
from torch.utils.data import Dataset

class _PlainView(Dataset):
    """Wrap a conditional dataset (spec, c, label) to expose (spec, label)."""
    def __init__(self, base_ds, indices):
        self.base = base_ds
        self.idxs = indices
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, i):
        spec, c, label = self.base[self.idxs[i]]
        return spec, label  # drop c so embed_data sees the same shape as before


def spec_to_tensor(x: np.ndarray) -> torch.Tensor:
    # same transform you used in training (module-level so workers could import, if needed)
    return torch.from_numpy(x).to(torch.float32).unsqueeze(0)


def latest_checkpoint(ckpt_dir: str) -> str:
    """Pick the newest 'final_*.pt' else newest 'ckpt_*.pt'."""
    finals = sorted(glob.glob(os.path.join(ckpt_dir, "final_*.pt")), key=os.path.getmtime)
    if finals:
        return finals[-1]
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")), key=os.path.getmtime)
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")


def strip_dataparallel_prefix(sd):
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }


# Gily's version - move this later outside
def load_gerbils_multi(gerbil_filepath, specs_per_file, families=[2],
                 test_size=0.2, seed=92, check=True):

    try:
        len(families)
    except Exception:
        families = [families]

    # Normalize gerbil_filepath into a "get_roots(family)" accessor
    if isinstance(gerbil_filepath, dict):
        def get_roots(fam):  # per-family explicit roots
            return gerbil_filepath.get(fam, [])
    elif isinstance(gerbil_filepath, (list, tuple)):
        def get_roots(fam):  # same set of roots for all families
            return list(gerbil_filepath)
    else:
        def get_roots(fam):  # single root (original)
            return [gerbil_filepath]

    specs_in_file = []
    all_family_specs = []
    all_family_ids = []

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
                        sif = len(f['specs'])
                        specs_in_file.append(sif)

        # accumulate
        all_family_specs += fam_spec_fns
        all_family_ids.append(ii * np.ones((len(fam_spec_fns),)))  # keep original labeling (0..K-1)

    if check and specs_in_file:
        num_specs = np.unique(specs_in_file)
        assert len(num_specs) == 1, print(f"Files have different numbers of specs in them! {num_specs}")
        if num_specs[0] != specs_per_file:
            print(f"expected {specs_per_file} specs per file, found {num_specs[0]}; updating")
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

def _make_ds_and_loaders(fns_tr, fns_te, ids_tr, ids_te, specs_per_file):
    # datasets
    train_ds_cond = bird_data(fns_tr, ids_tr, specs_per_file=specs_per_file,
                              transform=spec_to_tensor, conditional=True,  conditional_factor=COND_FACTOR)
    test_ds_cond  = bird_data(fns_te, ids_te, specs_per_file=specs_per_file,
                              transform=spec_to_tensor, conditional=True,  conditional_factor=COND_FACTOR)

    train_ds = bird_data(fns_tr, ids_tr, specs_per_file=specs_per_file,
                         transform=spec_to_tensor, conditional=False)
    test_ds  = bird_data(fns_te, ids_te, specs_per_file=specs_per_file,
                         transform=spec_to_tensor, conditional=False)

    # loaders
    train_loader_cond = DataLoader(train_ds_cond, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=True)
    test_loader_cond  = DataLoader(test_ds_cond,  batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    return (train_ds_cond, test_ds_cond, train_loader_cond, test_loader_cond,
            train_loader, test_loader)


def _bin_loaders(ds_cond, batch_size, num_workers, pin_memory=True):
    """From a (spec, c, label) dataset → dict{bin: DataLoader of (spec,label)}."""
    # pass once to gather indices per bin
    bin_to_idxs = {0: [], 1: [], 2: []}
    tmp = DataLoader(ds_cond, batch_size=128, shuffle=False, num_workers=0)
    idx = 0
    for _, c, _ in tmp:
        bins = c.argmax(dim=1).tolist()
        for b in bins:
            bin_to_idxs[b].append(idx)
            idx += 1

    # wrap each bin view into a (spec,label) dataset
    views = {b: _PlainView(ds_cond, bin_to_idxs[b]) for b in (0, 1, 2)}
    loaders = {
        b: DataLoader(views[b], batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=pin_memory)
        for b in (0, 1, 2)
    }
    return loaders


def build_loaders():
    (train_fns, test_fns), (train_ids, test_ids), specs_per_file = load_gerbils_multi(
        gerbil_filepath=DATA_ROOT,
        specs_per_file=SPECS_PER_FILE,
        families=TEST_FAMILY_IDS,
        test_size=TEST_SIZE,
        seed=SPLIT_SEED,
        check=True,
    )

    (train_ds_cond, test_ds_cond,
     train_loader_cond, test_loader_cond,
     train_loader, test_loader) = _make_ds_and_loaders(
        train_fns, test_fns, train_ids, test_ids, specs_per_file
    )

    # per-bin loaders for train/test (plain view: (spec,label))
    train_bin_loaders = _bin_loaders(train_ds_cond, BATCH_SIZE, NUM_WORKERS, pin_memory=True)
    test_bin_loaders  = _bin_loaders(test_ds_cond,  BATCH_SIZE, NUM_WORKERS, pin_memory=True)

    return (train_loader_cond, test_loader_cond,
            train_loader, test_loader,
            test_bin_loaders, train_bin_loaders,
            train_fns, test_fns)

# def build_loaders():
#     # From one folder
#     # (train_fns, test_fns), (train_ids, test_ids), specs_per_file = load_gerbils(
#     #     DATA_ROOT,
#     #     specs_per_file=SPECS_PER_FILE,
#     #     families=TEST_FAMILY_IDS,
#     #     test_size=TEST_SIZE,
#     #     seed=SPLIT_SEED,
#     #     check=True,
#     # )
#     # From multiple folders
#     (train_fns, test_fns), (train_ids, test_ids), specs_per_file = load_gerbils_multi(
#         gerbil_filepath=DATA_ROOT,
#         specs_per_file=SPECS_PER_FILE,
#         families=TEST_FAMILY_IDS,
#         test_size=TEST_SIZE,
#         seed=SPLIT_SEED,
#         check=True,
#     )
#
#     # gets one sample at a time by index via __getitem__ and knows how many samples exist via __len__, each item is (spec,family_id) or (spec,c,family_id)
#     test_ds_cond = bird_data(test_fns, test_ids, specs_per_file=specs_per_file, transform=spec_to_tensor, conditional=True, conditional_factor=COND_FACTOR)
#     train_ds_cond = bird_data(train_fns, train_ids, specs_per_file=specs_per_file, transform=spec_to_tensor, conditional=True, conditional_factor=COND_FACTOR)
#
#     test_ds = bird_data(test_fns, test_ids, specs_per_file=specs_per_file, transform=spec_to_tensor, conditional=False)
#     train_ds = bird_data(train_fns, train_ids, specs_per_file=specs_per_file, transform=spec_to_tensor, conditional=False)
#
#     # wraps the dataset to give mini-batches, splits by batch_size
#     train_loader_cond = DataLoader(train_ds_cond, batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS, pin_memory=True)
#     test_loader_cond  = DataLoader(test_ds_cond,  batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS, pin_memory=True)
#
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS, pin_memory=True)
#     test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS, pin_memory=True)
#
#     # ---- which example is from which bin? ----
#     bin_to_idxs = {0: [], 1: [], 2: []}
#     tmp = DataLoader(test_ds_cond, batch_size=128, shuffle=False, num_workers=0) # iterates the dataset one after the other
#     idx = 0
#     for _, c, _ in tmp:
#         # c: [B,3] one-hot; get bin ids
#         bins = c.argmax(dim=1).tolist() # the bin of each sample [2, 2, 1, 0, 2, ...]
#         for b in bins:
#             bin_to_idxs[b].append(idx) # vecotr of sample indices for each bin
#             idx += 1
#             # bin_to_idxs = {
#             #   0: [0, 7, 11, ...],
#             #   1: [2, 5, 8, ...],
#             #   2: [1, 3, 4, 6, 9, ...],
#             # }
#
#     # Per-bin PLAIN views (drop c), so embed_data sees (data,label)
#     test_bin_views = {
#         b: _PlainView(test_ds_cond, bin_to_idxs[b]) for b in (0, 1, 2)
#     }
#     test_bin_loaders = { # build dataLoader per bin
#         b: DataLoader(test_bin_views[b], batch_size=BATCH_SIZE, shuffle=False,
#                       num_workers=NUM_WORKERS, pin_memory=True)
#         for b in (0, 1, 2)
#     }
#
#     # ---- TRAIN: which example is from which bin? (mirror of your TEST block) ----
#     train_bin_to_idxs = {0: [], 1: [], 2: []}
#     tmp_tr = DataLoader(train_ds_cond, batch_size=128, shuffle=False, num_workers=0)
#     idx = 0
#     for _, c, _ in tmp_tr:
#         bins = c.argmax(dim=1).tolist()
#         for b in bins:
#             train_bin_to_idxs[b].append(idx)
#             idx += 1
#
#     train_bin_views = {b: _PlainView(train_ds_cond, train_bin_to_idxs[b]) for b in (0, 1, 2)}
#     train_bin_loaders = {
#         b: DataLoader(train_bin_views[b], batch_size=BATCH_SIZE, shuffle=False,
#                       num_workers=NUM_WORKERS, pin_memory=True)
#         for b in (0, 1, 2)
#     }
#
#     return (train_loader_cond, test_loader_cond,
#             train_loader, test_loader,
#             test_bin_loaders, train_bin_loaders,  # <— add this
#             train_fns, test_fns)



def rebuild_model(device):
    decoder = get_decoder_arch(dataset_name="gerbil_ava", latent_dim=LATENT_DIM,
                               arch="conditional_qmc",cond_dim=COND_DIM) # added for conditionals
    model = QMCLVM(latent_dim=LATENT_DIM, device=device, decoder=decoder)
    return model


def load_model_weights(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = strip_dataparallel_prefix(ckpt["model_state_dict"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] missing keys: {missing}\n[load] unexpected keys: {unexpected}")
    model.to(device)
    model.eval()
    # losses from training, useful for plotting alongside test
    train_losses = ckpt.get("losses", [])
    return ckpt, train_losses


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 1) latent grid + losses (match training)
    latent_grid = gen_fib_basis(m=M_FIB)  # (wrap with %1 if you visualize coordinates)
    qmc_loss_func = binary_evidence
    qmc_lp        = binary_lp

    # 2) dataset/loader (same split params as training)
    (train_loader_cond, test_loader_cond,
     train_loader, test_loader,
     test_bin_loaders, train_bin_loaders,
     train_fns, test_fns) = build_loaders()

    # 2.5) plot spectrograms in bins
    # =========================
    # DUMP: per-bin spectrogram PNGs (train & test), plus a SUM image per bin
    # =========================
    # We use the dataset bins (0/1/2) you already built, compute each item's mean frequency on-the-fly,
    # save the individual spectrograms with a y-axis in kHz from MIN_FREQ_HZ..MAX_FREQ_HZ,
    # draw a dashed line at the mean frequency, and create one summed spectrogram per bin.

    from data.bird_data import calc_energy_weighted_median_hz
    from data.bird_data import calc_mean_freq
    FREQ_AXIS = FREQ_AXIS_GLOBAL

    def _safe(text: str) -> str:
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))

    def _save_spec_png_linaxis(spec_2d: np.ndarray, out_png: str, mean_freq_hz: float,
                               y_min_hz: float, y_max_hz: float,
                               cmap="magma", dpi=180, figsize=(5.0, 4.0)):
        import matplotlib.pyplot as plt
        y0_khz, y1_khz = y_min_hz / 1000.0, y_max_hz / 1000.0
        extent = [0, spec_2d.shape[1], y0_khz, y1_khz]

        vmin = np.percentile(spec_2d, 5)
        vmax = np.percentile(spec_2d, 95)
        if vmin == vmax:
            vmin, vmax = float(spec_2d.min()), float(spec_2d.max())

        plt.figure(figsize=figsize, dpi=dpi)
        im = plt.imshow(
            spec_2d, origin="lower", aspect="auto", extent=extent,
            cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
        )
        plt.colorbar(im, pad=0.01, shrink=0.9)
        plt.axhline(mean_freq_hz / 1000.0, color="white", linestyle="--", linewidth=1.0, alpha=0.9)
        plt.xlabel("Time (frames)")
        plt.ylabel("Frequency (kHz)")
        plt.tight_layout()
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()

    def _resize_to(arr: np.ndarray, target_shape):
        F0, T0 = arr.shape
        Ft, Tt = target_shape
        if (F0, T0) == (Ft, Tt):
            return arr
        return zoom(arr, (Ft / max(F0, 1), Tt / max(T0, 1)), order=1)

    def _dump_for_split(split_name: str, bin_loaders: dict, out_root: str,
                        per_item_normalize=True, target_sum_shape=None):
        os.makedirs(out_root, exist_ok=True)
        for b in (0, 1, 2):
            bin_dir = os.path.join(out_root, f"bin_{b}")
            os.makedirs(bin_dir, exist_ok=True)

            # First pass: figure target sum shape (if not forced)
            shapes = []
            for specs, labels in bin_loaders[b]:
                # specs: [B, 1, F, T]
                F = specs.shape[2]
                T = specs.shape[3]
                shapes.append((F, T))
            if not shapes:
                print(f"[{split_name}] bin {b}: no items")
                continue

            if target_sum_shape is None:
                Ft = max(s[0] for s in shapes)
                Tt = max(s[1] for s in shapes)
                tgt = (Ft, Tt)
            else:
                tgt = target_sum_shape

            # Second pass: save individuals + accumulate SUM
            spec_sum = np.zeros(tgt, dtype=np.float32)
            running_idx = 0

            # Re-iterate (no shuffle) to align with file order
            for specs, labels in bin_loaders[b]:
                # specs: [B, 1, F, T]
                specs_np = specs.numpy()  # CPU tensors by default in DataLoader
                B = specs_np.shape[0]

                for i in range(B):
                    spec_2d = specs_np[i, 0]  # [F, T]
                    # compute stat frequency in Hz w.r.t. your linear freq axis
                    #mean_hz = float(calc_mean_freq(spec_2d, freq_axis=FREQ_AXIS))
                    est_hz = float(calc_energy_weighted_median_hz(spec_2d, freq_axis=FREQ_AXIS))

                    # save individual
                    name = f"{split_name}_b{b}_item_{running_idx:06d}.png"
                    out_png = os.path.join(bin_dir, _safe(name))
                    _save_spec_png_linaxis(
                        spec_2d, out_png, est_hz,
                        y_min_hz=MIN_FREQ_HZ, y_max_hz=MAX_FREQ_HZ,
                        cmap="magma", dpi=180, figsize=(5.0, 4.0)
                    )

                    # add to SUM (resize + optional per-item normalization)
                    r = _resize_to(spec_2d, tgt).astype(np.float32)
                    if per_item_normalize:
                        mx = r.max()
                        if mx > 0:
                            r = r / mx
                    spec_sum += r

                    running_idx += 1

            # write SUM image + raw npy
            sum_png = os.path.join(bin_dir, "SUM_spectrogram.png")
            _save_spec_png_linaxis(
                spec_sum, sum_png, mean_freq_hz=MIN_FREQ_HZ-1,  # line outside range (no visible line)
                y_min_hz=MIN_FREQ_HZ, y_max_hz=MAX_FREQ_HZ,
                cmap="magma", dpi=200, figsize=(6.0, 4.5)
            )
            np.save(os.path.join(bin_dir, "SUM_spectrogram.npy"), spec_sum)
            print(f"[{split_name}] bin {b}: saved individuals + SUM -> {bin_dir}")




    # 3) model + checkpoint
    ckpt_path = latest_checkpoint(CKPT_DIR)
    print("loading ckpt:", ckpt_path)
    model = rebuild_model(device)
    ckpt, train_losses = load_model_weights(model, ckpt_path, device)

    run_id  = ckpt.get("run_id", Path(ckpt_path).stem)
    out_dir = os.path.join(CKPT_DIR, f"eval_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    # Choose output roots under your checkpoint eval dir
    spec_out_train = os.path.join(out_dir, "spec_bins_train")
    spec_out_test = os.path.join(out_dir, "spec_bins_test")

    # Dump both splits (you can comment one out if you only want test)
    _dump_for_split("train", train_bin_loaders, spec_out_train,
                    per_item_normalize=True, target_sum_shape=None)
    _dump_for_split("test", test_bin_loaders, spec_out_test,
                    per_item_normalize=True, target_sum_shape=None)

    # 4) evaluation
    with torch.no_grad():
        test_losses = test_epoch(
            model,
            test_loader_cond,
            latent_grid.to(device),
            qmc_loss_func,
            conditional=True,
        )

    # 5) save eval artifacts
    np.save(os.path.join(out_dir, "test_losses.npy"), np.asarray(test_losses, np.float32))
    with open(os.path.join(out_dir, "eval_meta.json"), "w") as f:
        json.dump(
            {
                "ckpt": os.path.abspath(ckpt_path),
                "epoch": ckpt.get("epoch"),
                "n_test_batches": len(test_loader),
                "device": str(device),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )

    #---------------
    # 6) plots
    # ---------------
    # 6a) training vs test evidence
    qmc_train_plot(train_losses, test_losses, save_fn=os.path.join(out_dir_fig, "train_vs_test.png"), show=True)

    # 6b) decoder grid visualization
    class CondWrapped(nn.Module):
        def __init__(self, base_model: nn.Module, c_onehot: torch.Tensor):
            super().__init__()
            self.base = base_model
            self.c = c_onehot.to(base_model.device).to(torch.float32)
            self.device = base_model.device  # so your plot code still works

        def forward(self, z, mod=False, random=False):
            # delegate to the real model, always passing the fixed c
            return self.base(z, mod=mod, random=random, c=self.c)

    onehots = {
        0: torch.tensor([[1., 0., 0.]]),
        1: torch.tensor([[0., 1., 0.]]),
        2: torch.tensor([[0., 0., 1.]]),
    }

    # Wrap the model per bin
    model_b0 = CondWrapped(model, onehots[0])
    model_b1 = CondWrapped(model, onehots[1])
    model_b2 = CondWrapped(model, onehots[2])

    # Now call model_grid_plot three times
    with torch.no_grad():
        model_grid_plot(model_b0, n_samples_dim=20, origin="lower", cm="inferno",
                        show=False, fn=os.path.join(out_dir_fig, "decoder_grid_bin0.png"))

        model_grid_plot(model_b1, n_samples_dim=20, origin="lower", cm="inferno",
                        show=False, fn=os.path.join(out_dir_fig, "decoder_grid_bin1.png"))

        model_grid_plot(model_b2, n_samples_dim=20, origin="lower", cm="inferno",
                        show=False, fn=os.path.join(out_dir_fig, "decoder_grid_bin2.png"))

    # embedding per bin
    # --- helper: side-by-side plot for one bin ---

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
        plt.savefig(os.path.join(out_dir_fig, f"embeddings_train_test_bin{bin_id}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --- embed per bin (train + test), still using embed_data ---
    onehots = {
        0: torch.tensor([[1., 0., 0.]], device=device),
        1: torch.tensor([[0., 1., 0.]], device=device),
        2: torch.tensor([[0., 0., 1.]], device=device),
    }

    for b in (0, 1, 2):
        c_b = onehots[b]

        # TEST embeddings in the c=b slice
        emb_te, lab_te = model.embed_data(
            latent_grid.to(device),
            test_bin_loaders[b],  # yields (spec, label) for bin b
            binary_lp,
            embed_type="rqmc",
            n_samples=5,
            c=c_b,  # key: evaluate on the proper conditional slice
        )

        # TRAIN embeddings in the same slice
        emb_tr, lab_tr = model.embed_data(
            latent_grid.to(device),
            train_bin_loaders[b],
            binary_lp,
            embed_type="rqmc",
            n_samples=5,
            c=c_b,
        )

        # plot side-by-side for this bin
        plot_bin_train_test(emb_tr, emb_te, b, out_dir_fig)

    ## ------ DBG -------
    lin0 = model.decoder[0]  # Linear(in_features=4+3, out=64)
    W = lin0.weight.detach().cpu().numpy()  # [64, 7]
    W_basis = W[:, :4]  # for latent basis (2*latent_dim)
    W_c = W[:, 4:]  # for the 3 one-hot dims


    print("||W_basis||_F =", np.linalg.norm(W_basis))
    print("||W_c||_F     =", np.linalg.norm(W_c))
    print("rowwise ||W_c|| mean:", np.mean(np.linalg.norm(W_c, axis=1)))

    from data.bird_data import calc_mean_freq  # uses linear or  current weighting


    # define the helper *inside* main; decorator and def must have the same indent
    @torch.no_grad()
    def _grid_meanfreq_map(model, z_grid, c_onehot):
        X = model(z_grid, random=False, mod=False, c=c_onehot)  # [K,1,H,W]
        vals = []
        for i in range(X.shape[0]):
            spec = X[i, 0].detach().cpu().numpy()
            # pass the frequency axis so the result is in Hz
            vals.append(calc_mean_freq(spec, freq_axis=FREQ_AXIS))
        return np.asarray(vals, dtype=np.float64)

    # ----- build the three maps -----
    device = model.decoder[0].weight.device
    z_grid = gen_fib_basis(m=15).to(device)
    Z = (z_grid % 1).detach().cpu().numpy()

    onehots = {
        0: torch.tensor([[1., 0., 0.]], device=device),
        1: torch.tensor([[0., 1., 0.]], device=device),
        2: torch.tensor([[0., 0., 1.]], device=device),
    }

    m0 = _grid_meanfreq_map(model, z_grid, onehots[0])
    m1 = _grid_meanfreq_map(model, z_grid, onehots[1])
    m2 = _grid_meanfreq_map(model, z_grid, onehots[2])

    # plotting (shared colorbar)
    import matplotlib.colors as colors
    # after you have m0, m1, m2 in Hz
    vals_hz = [m0, m1, m2]
    vals_khz = [v / 1000.0 for v in vals_hz]
    titles = ["Bin 0 (<22 kHz)", "Bin 1 (22–25 kHz)", "Bin 2 (≥25 kHz)"]

    # shared color scale in kHz
    vmin_khz = min(map(np.min, vals_khz))
    vmax_khz = max(map(np.max, vals_khz))
    norm_khz = colors.Normalize(vmin=vmin_khz, vmax=vmax_khz)
    cmap = "viridis"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    marker_size = 120
    alpha = 0.98

    for ax, v_khz, title in zip(axes, vals_khz, titles):
        sc = ax.scatter(
            Z[:, 0], Z[:, 1],
            c=v_khz, s=marker_size, alpha=alpha,
            cmap=cmap, norm=norm_khz,
            edgecolors="none", linewidths=0,
            rasterized=True
        )
        ax.set_title(title)
        ax.set_xlim(0, 1);
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", "box")
        ax.set_xticks([]);
        ax.set_yticks([])

    # colorbar from the last scatter; all share same norm/cmap so this is fine
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
    cbar.set_label("Mean frequency (kHz)")

    fig.tight_layout()
    plt.savefig(os.path.join(out_dir_fig, "meanfreq_scatter_bins_big_khz.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ----- Non-conditional ------
    # # 6c) latent embeddings scatter
    # # Test data embedding
    # test_embeddings, test_labels = model.embed_data(
    #     latent_grid.to(device),
    #     test_loader,
    #     qmc_lp,
    #     embed_type="rqmc",
    #     n_samples=5,
    # )
    #
    # # Train data embedding
    # train_embeddings, train_labels = model.embed_data(
    #     latent_grid.to(device),
    #     train_loader,
    #     qmc_lp,
    #     embed_type="rqmc",
    #     n_samples=5,
    # )


    # =========================
    # Plot — all families together
    # =========================

   # fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    # Test
    # ax = axes[0]
    # ax.scatter(test_embeddings[:, 0], test_embeddings[:, 1],
    #            s=3, alpha=0.6, c="C0", zorder=1)
    # ax = format_plot_axis(
    #     ax, xlim=(0, 1), ylim=(0, 1),
    #     xlabel="Latent dim 1", ylabel="Latent dim 2",
    #     title="Test (all families pooled)"
    # )
    #
    # # Train
    # ax = axes[1]
    # ax.scatter(train_embeddings[:, 0], train_embeddings[:, 1],
    #            s=3,marker = ".", alpha=0.6, c="C0", zorder=1)
    # ax = format_plot_axis(
    #     ax, xlim=(0, 1), ylim=(0, 1),
    #     xlabel="Latent dim 1", ylabel="Latent dim 2",
    #     title="Train (all families pooled)"
    # )
    #
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir_fig, "embeddings_allfamilies_train_vs_test.png"),
    #             dpi=300, bbox_inches="tight")
    # plt.show();
    # plt.close(fig)

    # =========================
    # Plot - families seperately
    # =========================
    # ========= Per-family panels (same color), Train vs Test =========
    # families_present = sorted(np.unique(np.concatenate([train_labels, test_labels])))
    #
    # for fam in families_present:
    #     m_tr = (train_labels == fam)
    #     m_te = (test_labels == fam)
    #
    #     fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    #
    #     # Test (family = fam)
    #     ax = axes[0]
    #     ax.scatter(test_embeddings[m_te, 0], test_embeddings[m_te, 1],
    #                s=3,marker = ".", alpha=0.6, c="C0", zorder=1)
    #     ax = format_plot_axis(
    #         ax, xlim=(0, 1), ylim=(0, 1),
    #         xlabel="Latent dim 1", ylabel="Latent dim 2",
    #         title=f"Test — family {int(fam)}"
    #     )
    #
    #     # Train (family = fam)
    #     ax = axes[1]
    #     ax.scatter(train_embeddings[m_tr, 0], train_embeddings[m_tr, 1],
    #                s=3, marker = ".", alpha=0.6, c="C0", zorder=1)
    #     ax = format_plot_axis(
    #         ax, xlim=(0, 1), ylim=(0, 1),
    #         xlabel="Latent dim 1", ylabel="Latent dim 2",
    #         title=f"Train — family {int(fam)}"
    #     )
    #
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(out_dir_fig, f"embeddings_family{int(fam)}_train_vs_test.png"),
    #                 dpi=300, bbox_inches="tight")
    #     plt.show();
    #     plt.close(fig)

    # =========================
    # Plot - train/test seperately
    # =========================
    # import math
    #
    # def _grid_nrows_ncols(n, max_cols=4):
    #     cols = min(n, max_cols)
    #     rows = math.ceil(n / cols)
    #     return rows, cols
    #
    # families_present = sorted(np.unique(np.concatenate([train_labels, test_labels])))
    # n = len(families_present)
    # rows, cols = _grid_nrows_ncols(n, max_cols=4)
    #
    # # ---------- TRAIN panels ----------
    # fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), sharex=True, sharey=True)
    # axes = np.atleast_2d(axes)
    #
    # for idx, fam in enumerate(families_present):
    #     r, c = divmod(idx, cols)
    #     ax = axes[r, c]
    #     m_tr = (train_labels == fam)
    #     ax.scatter(train_embeddings[m_tr, 0], train_embeddings[m_tr, 1],
    #                s=3, marker=".", c="C0", alpha=0.6, linewidths=0)
    #     ax = format_plot_axis(
    #         ax, xlim=(0, 1), ylim=(0, 1),
    #         xlabel="Latent dim 1", ylabel="Latent dim 2",
    #         title=f"Train — family {int(fam)}"
    #     )
    #
    # # hide any unused axes
    # for j in range(n, rows * cols):
    #     r, c = divmod(j, cols)
    #     axes[r, c].axis("off")
    #
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir_fig, "embeddings_train_panels_by_family.png"),
    #             dpi=300, bbox_inches="tight")
    # plt.show();
    # plt.close(fig)
    #
    # # ---------- TEST panels ----------
    # fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), sharex=True, sharey=True)
    # axes = np.atleast_2d(axes)
    #
    # for idx, fam in enumerate(families_present):
    #     r, c = divmod(idx, cols)
    #     ax = axes[r, c]
    #     m_te = (test_labels == fam)
    #     ax.scatter(test_embeddings[m_te, 0], test_embeddings[m_te, 1],
    #                s=3, marker=".", c="C0", alpha=0.6, linewidths=0)
    #     ax = format_plot_axis(
    #         ax, xlim=(0, 1), ylim=(0, 1),
    #         xlabel="Latent dim 1", ylabel="Latent dim 2",
    #         title=f"Test — family {int(fam)}"
    #     )
    #
    # for j in range(n, rows * cols):
    #     r, c = divmod(j, cols)
    #     axes[r, c].axis("off")
    #
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir_fig, "embeddings_test_panels_by_family.png"),
    #             dpi=300, bbox_inches="tight")
    # plt.show();
    # plt.close(fig)
    #
    # # =========================
    # # Plot - By location
    # # =========================
    # # Read locations from hdf5 files
    # def _read_locations_for_file(h5_path):
    #     with h5py.File(h5_path, 'r') as f:
    #         locs = f['locations'][:]  # bytes array length = number of specs in this file
    #     # decode bytes -> str
    #     return np.array([x.decode('ASCII') for x in locs], dtype=object)
    #
    # def build_locations_vector(file_list):
    #     """
    #     Returns a 1D array of strings (arena_1/arena_2/underground) aligned
    #     with the order bird_data/test_loader iterate (file-major, no shuffle).
    #     """
    #     out = []
    #     for p in file_list:
    #         out.extend(_read_locations_for_file(p))
    #     return np.array(out, dtype=object)
    #
    # # locations
    # train_locs = build_locations_vector(train_fns)  # shape = len(train_embeddings)
    # test_locs = build_locations_vector(test_fns)  # shape = len(test_embeddings)
    #
    # #
    # # plot - one family at one location (train vs test panels)
    # def plot_family_location(fam, loc_name,
    #                          train_embeddings, train_labels, train_locs,
    #                          test_embeddings, test_labels, test_locs,
    #                          out_dir_fig):
    #     m_tr = (train_labels == fam) & (train_locs == loc_name)
    #     m_te = (test_labels == fam) & (test_locs == loc_name)
    #
    #     fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    #
    #     # Test
    #     ax = axes[0]
    #     ax.scatter(test_embeddings[m_te, 0], test_embeddings[m_te, 1],
    #                s=3, marker=".", alpha=0.7, c="C0")
    #     ax = format_plot_axis(
    #         ax, xlim=(0, 1), ylim=(0, 1),
    #         xlabel="Latent dim 1", ylabel="Latent dim 2",
    #         title=f"Test — family {int(fam)}, {loc_name}"
    #     )
    #
    #     # Train
    #     ax = axes[1]
    #     ax.scatter(train_embeddings[m_tr, 0], train_embeddings[m_tr, 1],
    #                s=3, marker=".", alpha=0.7, c="C0")
    #     ax = format_plot_axis(
    #         ax, xlim=(0, 1), ylim=(0, 1),
    #         xlabel="Latent dim 1", ylabel="Latent dim 2",
    #         title=f"Train — family {int(fam)}, {loc_name}"
    #     )
    #
    #     plt.tight_layout()
    #     fn = os.path.join(out_dir_fig, f"embeddings_family{int(fam)}_{loc_name}_train_vs_test.png")
    #     plt.savefig(fn, dpi=300, bbox_inches="tight")
    #     plt.close(fig)
    #
    # for fam in sorted(np.unique(np.concatenate([train_labels, test_labels]))):
    #     for loc in ["arena_1", "arena_2", "underground"]:
    #         plot_family_location(fam, loc,
    #                                  train_embeddings, train_labels, train_locs,
    #                                  test_embeddings, test_labels, test_locs,
    #                                  out_dir_fig)
    #
    # # plot - “all families pooled” but filtered to a single location
    # def plot_all_families_at_location(loc_name,
    #                                   train_embeddings, train_locs,
    #                                   test_embeddings, test_locs,
    #                                   out_dir_fig):
    #     m_tr = (train_locs == loc_name)
    #     m_te = (test_locs == loc_name)
    #
    #     fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    #
    #     # Test
    #     ax = axes[0]
    #     ax.scatter(test_embeddings[m_te, 0], test_embeddings[m_te, 1],
    #                s=3, marker=".", alpha=0.7, c="C0")
    #     ax = format_plot_axis(
    #         ax, xlim=(0, 1), ylim=(0, 1),
    #         xlabel="Latent dim 1", ylabel="Latent dim 2",
    #         title=f"Test — all families @ {loc_name}"
    #     )
    #
    #     # Train
    #     ax = axes[1]
    #     ax.scatter(train_embeddings[m_tr, 0], train_embeddings[m_tr, 1],
    #                s=3, marker=".", alpha=0.7, c="C0")
    #     ax = format_plot_axis(
    #         ax, xlim=(0, 1), ylim=(0, 1),
    #         xlabel="Latent dim 1", ylabel="Latent dim 2",
    #         title=f"Train — all families @ {loc_name}"
    #     )
    #
    #     plt.tight_layout()
    #     fn = os.path.join(out_dir_fig, f"embeddings_allfamilies_{loc_name}_train_vs_test.png")
    #     plt.savefig(fn, dpi=300, bbox_inches="tight")
    #     plt.close(fig)
    #
    # for loc in ["arena_1", "arena_2", "underground"]:
    #     plot_all_families_at_location(loc, train_embeddings, train_locs,
    #                                   test_embeddings, test_locs,
    #                                   out_dir_fig)
    #
    # # =========================
    # Plot — embeddings over the decoded grid background
    # (place this AFTER train/test embeddings are computed)
    # =========================
    # # same EPS as your visualize module uses
    # EPS1 = 1e-15
    # EPS2 = 1e-6
    #
    # def _embed_to_cell_idx(x, y, n, eps1=EPS1, eps2=EPS2):
    #     # map [0,1] coords to a cell index matching "sample {i*n + j}"
    #     u = np.clip((x - eps1) / max(1e-12, (1 - eps1 - eps2)), 0.0, 0.999999)
    #     v = np.clip((y - eps1) / max(1e-12, (1 - eps1 - eps2)), 0.0, 0.999999)
    #     i = int(u * n)  # column
    #     j = int(v * n)  # row
    #     return i * n + j
    #
    # def _overlay_embeddings_on_grid(fig, axes_map, emb_xy, n, dot="•", color="white"):
    #     if emb_xy.size == 0:
    #         return
    #     idxs = [_embed_to_cell_idx(float(x), float(y), n) for x, y in emb_xy]
    #     for idx in sorted(set(idxs)):
    #         ax = axes_map.get(f"sample {idx}")
    #         if ax is not None:
    #             ax.text(0.92, 0.88, dot, transform=ax.transAxes,
    #                     color=color, fontsize=10, fontweight="bold", zorder=10)
    #
    # GRID_N = 20
    # emb_all = np.vstack([train_embeddings, test_embeddings])
    #
    # with torch.no_grad():
    #     fig, axmap = model_grid_plot(
    #         model,
    #         n_samples_dim=GRID_N,
    #         origin="lower",
    #         cm="inferno",
    #         show=False,
    #         return_fig=True,  # <-- new
    #     )
    #
    # _overlay_embeddings_on_grid(fig, axmap, emb_all, GRID_N, color="white")
    # fig.suptitle("All families — usage over decoded latent grid", y=0.98, fontsize=14)
    # fig.tight_layout()
    # fig.savefig(os.path.join(out_dir_fig, "overlay_allfamilies_on_grid.png"),
    #             dpi=300, bbox_inches="tight")
    # plt.close(fig)


# trial 2
# --- Overlay on latent lattice (monochrome) ---
# latent_bg = (latent_grid % 1).detach().cpu().numpy()
#
# for fam in sorted(np.unique(np.concatenate([train_labels, test_labels]))):
#     m_tr = (train_labels == fam)
#     m_te = (test_labels == fam)
#
#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.set_facecolor("white")
#
#     # lattice: darker gray so white points pop
#     ax.scatter(
#         latent_bg[:, 0], latent_bg[:, 1],
#         s=3, marker=".", c="0.6", alpha=1.0, linewidths=0, zorder=0
#     )
#
#     # TEST: hollow white with black edge
#     ax.scatter(
#         test_embeddings[m_te, 0], test_embeddings[m_te, 1],
#         s=10, facecolors="white", edgecolors="black", linewidths=0.6, zorder=2, label="test"
#     )
#
#     # TRAIN: solid white with thin black edge
#     ax.scatter(
#         train_embeddings[m_tr, 0], train_embeddings[m_tr, 1],
#         s=6, facecolors="white", edgecolors="black", linewidths=0.4, zorder=3, label="train"
#     )
#
#     ax = format_plot_axis(
#         ax, xlim=(0, 1), ylim=(0, 1),
#         xlabel="Latent dim 1", ylabel="Latent dim 2",
#         title=f"Family {int(fam)} — train/test over lattice (mono)"
#     )
#     ax.legend(frameon=False, loc="best")
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir_fig, f"embeddings_family{int(fam)}_over_grid_mono2.png"),
#                 dpi=300, bbox_inches="tight")
#     plt.show(); plt.close(fig)





    # Old test plot:
    # ax = plt.gca()
    # ax.scatter(test_embeddings[:, 0], test_embeddings[:, 1], s=1, alpha=0.5)
    # ax = format_plot_axis(
    #     ax,
    #     xlim=(0, 1),
    #     ylim=(0, 1),
    #     xlabel="Latent dim 1",
    #     ylabel="Latent dim 2",
    #     title="Latent embeddings of test dataset",
    # )
    # plt.show()
    # plt.close()
    #
    # print(f"Saved eval outputs → {out_dir}")

    ################################################################################
    # Miles' plot code:
    ################################################################################

    # # both plots overlaid (train+test):
    # ax = plt.gca()
    #
    # # Base layer: TRAIN (light gray, behind)
    # ax.scatter(
    #     train_embeddings[:, 0], train_embeddings[:, 1],
    #     s=4, c="0.7", alpha=0.5, label="test", zorder=1
    # )
    #
    # # Overlay: TEST (outlined points, on top)
    # ax.scatter(
    #     test_embeddings[:, 0], test_embeddings[:, 1],
    #     s=8, facecolors="none", edgecolors="C3", linewidth=0.6,
    #     label="train", zorder=2
    # )
    #
    # ax = format_plot_axis(
    #     ax,
    #     xlim=(0, 1),
    #     ylim=(0, 1),
    #     xlabel="Latent dim 1",
    #     ylabel="Latent dim 2",
    #     title="Latent embeddings (train gray, test red)",
    # )
    #
    # ax.legend(frameon=False, loc="best")
    # plt.show()
    # plt.close()


if __name__ == "__main__":
    # Windows-safe entrypoint (even though eval uses num_workers=0)
    import torch.multiprocessing as mp
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
