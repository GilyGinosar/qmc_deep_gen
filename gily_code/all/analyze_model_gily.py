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
    1: [r"D:\Data\235",r"D:\Data\237"],
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
COND_FACTOR    = "rule3_bands"   # the one-hot(3) we added in bird_data
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
def load_gerbils_multi(gerbil_filepath, specs_per_file, families=[1],
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



# ------------------------
# New
#-----------------
# ---
# Helpers for embedding by groups (per-bin)
# ---

def _gather_bin_indices(ds_cond) -> dict[int, list[int]]:
    """
    Iterate a (spec, c, label) dataset in file-major order and return
    {bin_id: [global item indices]} with no shuffling.
    """
    bin_to_idxs = {0: [], 1: [], 2: []}
    tmp = DataLoader(ds_cond, batch_size=128, shuffle=False, num_workers=0)
    idx = 0
    for _, c, _ in tmp:                # c is one-hot [B,3]
        bins = c.argmax(dim=1).tolist()
        for b in bins:
            bin_to_idxs[b].append(idx)
            idx += 1
    return bin_to_idxs

def _read_locations_for_file(h5_path):
    with h5py.File(h5_path, 'r') as f:
        locs = f['locations'][:]  # bytes array, len = specs_per_file
    return np.array([x.decode('ASCII') for x in locs], dtype=object)

def build_locations_vector(file_list):
    """
    Returns a 1D array of strings aligned with the order that bird_data
    (no shuffle) iterates: concatenate per-file locations in file-major order.
    """
    out = []
    for p in file_list:
        out.extend(_read_locations_for_file(p))
    return np.array(out, dtype=object)

def _loc_bucketize(loc_str_array: np.ndarray) -> np.ndarray:
    """Map location strings -> {'arena', 'underground'}."""
    s = np.char.lower(loc_str_array.astype(str))
    arena_mask = np.logical_or(np.char.find(s, "arena_1") >= 0,
                               np.char.find(s, "arena_2") >= 0)
    out = np.where(arena_mask, "arena", "underground")
    return out

def _nice_colors(n):
    base = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
    if n <= len(base):
        return base[:n]
    return [base[i % len(base)] for i in range(n)]

def _plot_triptych_by_groups(emb_xy: np.ndarray,
                             groups: np.ndarray,
                             group_order: list,
                             group_display: list,
                             colors: list,
                             title_prefix: str,
                             split_name: str,
                             bin_id: int,
                             out_dir: str,
                             file_tag: str):
    """
    Make 3 panels for one split/bin:
      (1) combined overlay (all groups, colored)
      (2) group A only
      (3) group B only
    If there are >2 groups (families), panel (2/3) show first two groups by order.
    """
    if emb_xy.size == 0:
        return

    # --- (1) combined overlay ---
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    for name, col in zip(group_order, colors):
        m = (groups == name)
        if np.any(m):
            ax.scatter(emb_xy[m, 0], emb_xy[m, 1], s=6, marker=".", alpha=0.9, c=col, label=str(name))
    format_plot_axis(ax, xlim=(0,1), ylim=(0,1),
                     xlabel="Latent dim 1", ylabel="Latent dim 2",
                     title=f"{title_prefix} — {split_name} — bin {bin_id}")
    ax.legend(frameon=False, loc="best", markerscale=1.6)
    plt.tight_layout()
    fn = os.path.join(out_dir, f"{file_tag}_combined_{split_name}_bin{bin_id}.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # choose up to two groups for the separate panels
    gA = group_order[0] if len(group_order) > 0 else None
    gB = group_order[1] if len(group_order) > 1 else None
    colA = colors[0] if len(colors) > 0 else "C0"
    colB = colors[1] if len(colors) > 1 else "C1"

    # --- (2) separate panel for group A ---
    if gA is not None:
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        m = (groups == gA)
        if np.any(m):
            ax.scatter(emb_xy[m, 0], emb_xy[m, 1], s=7, marker=".", alpha=0.95, c=colA)
        format_plot_axis(ax, xlim=(0,1), ylim=(0,1),
                         xlabel="Latent dim 1", ylabel="Latent dim 2",
                         title=f"{title_prefix}: {group_display[0]} — {split_name} — bin {bin_id}")
        plt.tight_layout()
        fn = os.path.join(out_dir, f"{file_tag}_onlyA_{split_name}_bin{bin_id}.png")
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --- (3) separate panel for group B ---
    if gB is not None:
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        m = (groups == gB)
        if np.any(m):
            ax.scatter(emb_xy[m, 0], emb_xy[m, 1], s=7, marker=".", alpha=0.95, c=colB)
        format_plot_axis(ax, xlim=(0,1), ylim=(0,1),
                         xlabel="Latent dim 1", ylabel="Latent dim 2",
                         title=f"{title_prefix}: {group_display[1]} — {split_name} — bin {bin_id}")
        plt.tight_layout()
        fn = os.path.join(out_dir, f"{file_tag}_onlyB_{split_name}_bin{bin_id}.png")
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        plt.close(fig)

def plot_family_triptych(emb_xy: np.ndarray,
                         fam_labels: np.ndarray,
                         split_name: str,
                         bin_id: int,
                         out_dir: str):
    fams = list(np.unique(fam_labels))
    colors = _nice_colors(len(fams))
    display = [f"family {int(f)}" for f in fams]
    _plot_triptych_by_groups(emb_xy, fam_labels, fams, display, colors,
                             title_prefix="Embeddings by family",
                             split_name=split_name, bin_id=bin_id,
                             out_dir=out_dir, file_tag="emb_by_family")

def plot_location_triptych(emb_xy: np.ndarray,
                           loc_bucket: np.ndarray,
                           split_name: str,
                           bin_id: int,
                           out_dir: str):
    order = ["arena", "underground"]
    colors = ["C0", "C3"]
    display = ["arena", "underground"]
    _plot_triptych_by_groups(emb_xy, loc_bucket, order, display, colors,
                             title_prefix="Embeddings by location",
                             split_name=split_name, bin_id=bin_id,
                             out_dir=out_dir, file_tag="emb_by_location")






# -------------------------------------------------------------------------------


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

    # 2.5 for embedding
    # Indices per bin in dataset order (no shuffle)
    train_bin_to_idxs = _gather_bin_indices(train_loader_cond.dataset)
    test_bin_to_idxs = _gather_bin_indices(test_loader_cond.dataset)

    # Locations aligned to dataset order
    train_locs_all = build_locations_vector(train_fns)  # len = #items in train set
    test_locs_all = build_locations_vector(test_fns)

    FREQ_AXIS = FREQ_AXIS_GLOBAL


    # 3) model + checkpoint
    ckpt_path = latest_checkpoint(CKPT_DIR)
    print("loading ckpt:", ckpt_path)
    model = rebuild_model(device)
    ckpt, train_losses = load_model_weights(model, ckpt_path, device)

    run_id  = ckpt.get("run_id", Path(ckpt_path).stem)
    out_dir = os.path.join(CKPT_DIR, f"eval_{run_id}")
    os.makedirs(out_dir, exist_ok=True)




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
                "n_test_batches": len(test_loader_cond),
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

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.base, name)

        @torch.no_grad()
        def forward(self, z, mod=False, random=False):
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
    #
    # def plot_bin_train_test(emb_train, emb_test, bin_id, out_dir):
    #     fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    #
    #     ax = axes[0]
    #     ax.scatter(emb_test[:, 0], emb_test[:, 1], s=3, marker=".", alpha=0.7, c="C0")
    #     format_plot_axis(ax, xlim=(0, 1), ylim=(0, 1),
    #                      xlabel="Latent dim 1", ylabel="Latent dim 2",
    #                      title=f"Bin {bin_id} — Test")
    #
    #     ax = axes[1]
    #     ax.scatter(emb_train[:, 0], emb_train[:, 1], s=3, marker=".", alpha=0.7, c="C0")
    #     format_plot_axis(ax, xlim=(0, 1), ylim=(0, 1),
    #                      xlabel="Latent dim 1", ylabel="Latent dim 2",
    #                      title=f"Bin {bin_id} — Train")
    #
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(out_dir_fig, f"embeddings_train_test_bin{bin_id}.png"),
    #                 dpi=300, bbox_inches="tight")
    #     plt.close(fig)
    #
    # # --- embed per bin (train + test), still using embed_data ---
    # onehots = {
    #     0: torch.tensor([[1., 0., 0.]], device=device),
    #     1: torch.tensor([[0., 1., 0.]], device=device),
    #     2: torch.tensor([[0., 0., 1.]], device=device),
    # }
    #
    # all_emb_tr, all_lab_tr, all_loc_tr = [], [], []
    # all_emb_te, all_lab_te, all_loc_te = [], [], []
    #
    # # ---- helper for readable crosstabs ----
    # def _crosstab_counts(a, b):
    #     from collections import Counter
    #     a = a.tolist() if hasattr(a, "tolist") else list(a)
    #     b = b.tolist() if hasattr(b, "tolist") else list(b)
    #     ct = Counter(zip(a, b))
    #     rows = sorted(set(a))
    #     cols = sorted(set(b))
    #     return rows, cols, ct
    #
    #
    # for b in (0, 1, 2):
    #     c_b = onehots[b]
    #
    #     # TEST embeddings in the c=b slice
    #     emb_te, lab_te = model.embed_data(
    #         latent_grid.to(device),
    #         test_bin_loaders[b],  # yields (spec, label) for bin b
    #         binary_lp,
    #         embed_type="rqmc",
    #         n_samples=5,
    #         c=c_b,  # key: evaluate on the proper conditional slice
    #     )
    #
    #     # TRAIN embeddings in the same slice
    #     emb_tr, lab_tr = model.embed_data(
    #         latent_grid.to(device),
    #         train_bin_loaders[b],
    #         binary_lp,
    #         embed_type="rqmc",
    #         n_samples=5,
    #         c=c_b,
    #     )
    #
    #     # plot side-by-side for this bin
    #     plot_bin_train_test(emb_tr, emb_te, b, out_dir_fig)
    #
    #     # ---- per-bin metadata slices (aligned with dataset order) ----
    #     idx_tr_b = np.array(train_bin_to_idxs[b], dtype=int)
    #     idx_te_b = np.array(test_bin_to_idxs[b], dtype=int)
    #
    #     loc_tr_b = _loc_bucketize(train_locs_all[idx_tr_b])
    #     loc_te_b = _loc_bucketize(test_locs_all[idx_te_b])
    #
    #     # ---- SANITY CHECKS (before embedding plots) ----
    #     print(f"\n[bin {b}] TRAIN uniques — families: {np.unique(lab_tr)}, locations: {np.unique(loc_tr_b)}")
    #     rows, cols, ct = _crosstab_counts(lab_tr, loc_tr_b)
    #     print(f"[bin {b}] TRAIN crosstab (family x location). cols={cols}")
    #     for r in rows:
    #         row = [ct.get((r, c), 0) for c in cols]
    #         print(f"  fam {r}: {row}")
    #
    #     print(f"[bin {b}] TEST  uniques — families: {np.unique(lab_te)}, locations: {np.unique(loc_te_b)}")
    #     rows, cols, ct = _crosstab_counts(lab_te, loc_te_b)
    #     print(f"[bin {b}] TEST  crosstab (family x location). cols={cols}")
    #     for r in rows:
    #         row = [ct.get((r, c), 0) for c in cols]
    #         print(f"  fam {r}: {row}")
    #
    #     # ---- per-bin triptychs: families (train & test) ----
    #     plot_family_triptych(emb_tr, lab_tr, split_name="train", bin_id=b, out_dir=out_dir_fig)
    #     plot_family_triptych(emb_te, lab_te, split_name="test", bin_id=b, out_dir=out_dir_fig)
    #
    #     # ---- per-bin triptychs: locations (train & test) ----
    #     plot_location_triptych(emb_tr, loc_tr_b, split_name="train", bin_id=b, out_dir=out_dir_fig)
    #     plot_location_triptych(emb_te, loc_te_b, split_name="test", bin_id=b, out_dir=out_dir_fig)
    #


    # -------------------------
    # Plots
    # -------------------------
    #
    #
    #
    #
    # lin0 = model.decoder[0]  # Linear(in_features=4+3, out=64)
    # W = lin0.weight.detach().cpu().numpy()  # [64, 7]
    # W_basis = W[:, :4]  # for latent basis (2*latent_dim)
    # W_c = W[:, 4:]  # for the 3 one-hot dims
    #
    #
    # print("||W_basis||_F =", np.linalg.norm(W_basis))
    # print("||W_c||_F     =", np.linalg.norm(W_c))
    # print("rowwise ||W_c|| mean:", np.mean(np.linalg.norm(W_c, axis=1)))
    #
    # from data.bird_data import calc_mean_freq  # uses linear or  current weighting
    #
    #
    # # define the helper *inside* main; decorator and def must have the same indent
    # @torch.no_grad()
    # def _grid_meanfreq_map(model, z_grid, c_onehot):
    #     X = model(z_grid, random=False, mod=False, c=c_onehot)  # [K,1,H,W]
    #     vals = []
    #     for i in range(X.shape[0]):
    #         spec = X[i, 0].detach().cpu().numpy()
    #         # pass the frequency axis so the result is in Hz
    #         vals.append(calc_mean_freq(spec, freq_axis=FREQ_AXIS))
    #     return np.asarray(vals, dtype=np.float64)
    #
    # # ----- build the three maps -----
    # device = model.decoder[0].weight.device
    # z_grid = gen_fib_basis(m=15).to(device)
    # Z = (z_grid % 1).detach().cpu().numpy()
    #
    # onehots = {
    #     0: torch.tensor([[1., 0., 0.]], device=device),
    #     1: torch.tensor([[0., 1., 0.]], device=device),
    #     2: torch.tensor([[0., 0., 1.]], device=device),
    # }
    #
    # m0 = _grid_meanfreq_map(model, z_grid, onehots[0])
    # m1 = _grid_meanfreq_map(model, z_grid, onehots[1])
    # m2 = _grid_meanfreq_map(model, z_grid, onehots[2])
    #
    # # plotting (shared colorbar)
    # import matplotlib.colors as colors
    # # after you have m0, m1, m2 in Hz
    # vals_hz = [m0, m1, m2]
    # vals_khz = [v / 1000.0 for v in vals_hz]
    # titles = ["Bin 0 (<22 kHz)", "Bin 1 (22–25 kHz)", "Bin 2 (≥25 kHz)"]
    #
    # # shared color scale in kHz
    # vmin_khz = min(map(np.min, vals_khz))
    # vmax_khz = max(map(np.max, vals_khz))
    # norm_khz = colors.Normalize(vmin=vmin_khz, vmax=vmax_khz)
    # cmap = "viridis"
    #
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    # marker_size = 120
    # alpha = 0.98
    #
    # for ax, v_khz, title in zip(axes, vals_khz, titles):
    #     sc = ax.scatter(
    #         Z[:, 0], Z[:, 1],
    #         c=v_khz, s=marker_size, alpha=alpha,
    #         cmap=cmap, norm=norm_khz,
    #         edgecolors="none", linewidths=0,
    #         rasterized=True
    #     )
    #     ax.set_title(title)
    #     ax.set_xlim(0, 1);
    #     ax.set_ylim(0, 1)
    #     ax.set_aspect("equal", "box")
    #     ax.set_xticks([]);
    #     ax.set_yticks([])
    #
    # # colorbar from the last scatter; all share same norm/cmap so this is fine
    # cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
    # cbar.set_label("Mean frequency (kHz)")
    #
    # fig.tight_layout()
    # plt.savefig(os.path.join(out_dir_fig, "meanfreq_scatter_bins_big_khz.png"),
    #             dpi=300, bbox_inches="tight")
    # plt.close(fig)




if __name__ == "__main__":
    # Windows-safe entrypoint (even though eval uses num_workers=0)
    import torch.multiprocessing as mp
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
