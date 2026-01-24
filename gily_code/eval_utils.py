"""
Shared helpers for evaluation scripts.
- Data loading and splitting (conditional and plain).
- Checkpoint discovery and safe state loading.
- Simple bin utilities (argmax over one-hot c to group items).

"""

from __future__ import annotations

import glob
import os
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.bird_data import bird_data



# -----------------------
# Generic helpers
# -----------------------


def spec_to_tensor(x: np.ndarray) -> torch.Tensor:
    """Convert a numpy spectrogram (F, T) to a float32 tensor with channel dim."""
    return torch.from_numpy(x).to(torch.float32).unsqueeze(0)


def latest_checkpoint(ckpt_dir: str) -> str:
    """
    Return the newest checkpoint path in a directory.
    Prefers final_*.pt; falls back to ckpt_*.pt.
    """
    finals = sorted(glob.glob(os.path.join(ckpt_dir, "final_*.pt")), key=os.path.getmtime)
    if finals:
        return finals[-1]
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")), key=os.path.getmtime)
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")


def strip_dataparallel_prefix(sd: dict) -> dict:
    """Remove a leading 'module.' (DataParallel) from state_dict keys."""
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def load_model_weights(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    """
    loading checkpoint from `ckpt_path`
    Loads full pickle, grabs model_state_dict if present, otherwise treats blob as state_dict.
    Returns (train_losses, epoch).
    """
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        state = blob["model_state_dict"]
        train_losses = blob.get("losses", [])
        epoch = blob.get("epoch")
    else:
        state = blob
        train_losses = []
        epoch = None

    state = strip_dataparallel_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] missing keys: {missing}\n[load] unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    return train_losses, epoch


# -----------------------
# Data loading helpers
# -----------------------


def _normalize_families(families) -> List[int]:
    try:
        return list(families)
    except Exception:
        return [families]


def _roots_getter(gerbil_filepath):
    if isinstance(gerbil_filepath, dict):
        def get_roots(fam):  # type: ignore
            return gerbil_filepath.get(fam, [])
    elif isinstance(gerbil_filepath, (list, tuple)):
        def get_roots(fam):  # type: ignore
            return list(gerbil_filepath)
    else:
        def get_roots(fam):  # type: ignore
            return [gerbil_filepath]
    return get_roots


def load_gerbils_multi(
    gerbil_filepath,
    specs_per_file: int,
    families: Iterable[int],
    *,
    test_size: float = 0.2,
    seed: int = 92,
    check: bool = True,
) -> Tuple[Tuple[List[str], List[str]], Tuple[np.ndarray, np.ndarray], int]:
    """
    Collect HDF5 spectrogram files across families and split into train/test.

    gerbil_filepath: dict {fam: [roots]}, list/tuple of roots, or single root.
    Returns ((train_files, test_files), (train_family_ids, test_family_ids), specs_per_file)
    """
    families = _normalize_families(families)
    get_roots = _roots_getter(gerbil_filepath)

    specs_in_file: List[int] = []
    all_family_specs: List[str] = []
    all_family_ids: List[np.ndarray] = []

    for ii, family in enumerate(families):
        print(f"[load] family{family}")
        roots = get_roots(family)

        fam_spec_fns: List[str] = []
        for root in roots:
            spec_dir = os.path.join(root, "processed-data", f"family{family}")
            spec_fns = glob.glob(os.path.join(spec_dir, "*.hdf5"))
            fam_spec_fns.extend(spec_fns)

            if check:
                for spec_fn in tqdm(spec_fns, total=len(spec_fns), desc=f"checking {spec_dir}"):
                    with h5py.File(spec_fn, "r") as f:
                        specs_in_file.append(len(f["specs"]))

        all_family_specs += fam_spec_fns
        all_family_ids.append(ii * np.ones((len(fam_spec_fns),)))

    if check and specs_in_file:
        num_specs = np.unique(specs_in_file)
        assert len(num_specs) == 1, f"Files have different numbers of specs: {num_specs}"
        if num_specs[0] != specs_per_file:
            print(f"[warn] expected {specs_per_file}, found {num_specs[0]}; updating")
            specs_per_file = int(num_specs[0])

    all_family_ids_arr = np.hstack(all_family_ids) if len(all_family_ids) else np.array([])

    if test_size > 0 and len(all_family_specs) > 0:
        train_fns, test_fns, train_ids, test_ids = train_test_split(
            all_family_specs, all_family_ids_arr, test_size=test_size, random_state=seed
        )
    else:
        train_fns = test_fns = all_family_specs
        train_ids = test_ids = all_family_ids_arr

    return (train_fns, test_fns), (train_ids, test_ids), specs_per_file


def build_conditional_loaders(
    gerbil_roots,
    families: Iterable[int],
    *,
    specs_per_file: int,
    batch_size: int,
    num_workers: int,
    test_size: float,
    split_seed: int,
    conditional_factor: str,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, Tuple[List[str], List[str]]]:
    """
    Build conditional datasets/loaders (spec, c, label) plus plain datasets/loaders (spec, label).
    Returns (train_loader_cond, test_loader_cond, train_loader_plain, test_loader_plain, (train_files, test_files)).
    """
    (train_fns, test_fns), (train_ids, test_ids), specs_per_file = load_gerbils_multi(
        gerbil_filepath=gerbil_roots,
        specs_per_file=specs_per_file,
        families=families,
        test_size=test_size,
        seed=split_seed,
        check=True,
    )

    train_ds_cond = bird_data(
        train_fns,
        train_ids,
        specs_per_file=specs_per_file,
        transform=spec_to_tensor,
        conditional=True,
        conditional_factor=conditional_factor,
    )
    test_ds_cond = bird_data(
        test_fns,
        test_ids,
        specs_per_file=specs_per_file,
        transform=spec_to_tensor,
        conditional=True,
        conditional_factor=conditional_factor,
    )

    train_ds_plain = bird_data(
        train_fns, train_ids, specs_per_file=specs_per_file, transform=spec_to_tensor, conditional=False
    )
    test_ds_plain = bird_data(
        test_fns, test_ids, specs_per_file=specs_per_file, transform=spec_to_tensor, conditional=False
    )

    pin = bool(torch.cuda.is_available())
    train_loader_cond = DataLoader(train_ds_cond, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader_cond = DataLoader(test_ds_cond, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    train_loader_plain = DataLoader(train_ds_plain, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader_plain = DataLoader(test_ds_plain, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    return train_loader_cond, test_loader_cond, train_loader_plain, test_loader_plain, (train_fns, test_fns)


def build_plain_loaders(
    gerbil_roots,
    families: Iterable[int],
    *,
    specs_per_file: int,
    batch_size: int,
    num_workers: int,
    test_size: float,
    split_seed: int,
) -> Tuple[DataLoader, DataLoader, Tuple[List[str], List[str]]]:
    """Plain (unconditional) loaders for train/test."""
    (train_fns, test_fns), (train_ids, test_ids), specs_per_file = load_gerbils_multi(
        gerbil_filepath=gerbil_roots,
        specs_per_file=specs_per_file,
        families=families,
        test_size=test_size,
        seed=split_seed,
        check=True,
    )

    train_ds = bird_data(train_fns, train_ids, specs_per_file=specs_per_file, transform=spec_to_tensor, conditional=False)
    test_ds = bird_data(test_fns, test_ids, specs_per_file=specs_per_file, transform=spec_to_tensor, conditional=False)

    pin = bool(torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    return train_loader, test_loader, (train_fns, test_fns)


# -----------------------
# Bin utilities (for one-hot conditional factors)
# -----------------------


class _PlainView(Dataset):
    """Expose (spec, label) from a conditional dataset that yields (spec, c, label)."""

    def __init__(self, base_ds: Dataset, indices: List[int]):
        self.base = base_ds
        self.idxs = indices

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        spec, c, label = self.base[self.idxs[i]]
        return spec, label


def gather_bin_indices(ds_cond: Dataset, n_bins: int = 3) -> Dict[int, List[int]]:
    """
    Iterate a conditional dataset (no shuffle) and return {bin_id: [global idx]}.
    Assumes c is one-hot and uses argmax.
    """
    bin_to_idxs: Dict[int, List[int]] = {b: [] for b in range(n_bins)}
    tmp = DataLoader(ds_cond, batch_size=256, shuffle=False, num_workers=0)
    idx = 0
    for _, c, _ in tmp:
        bins = c.argmax(dim=1).tolist()
        for b in bins:
            bin_to_idxs[int(b)].append(idx)
            idx += 1
    return bin_to_idxs


def build_bin_loaders_from_cond(
    ds_cond: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    n_bins: int = 3,
) -> Dict[int, DataLoader]:
    """
    Create per-bin loaders returning (spec, label) using a conditional dataset
    that yields (spec, c, label). Uses argmax over c to decide bin.
    """
    bin_to_idxs = gather_bin_indices(ds_cond, n_bins=n_bins)
    pin = bool(torch.cuda.is_available())
    loaders = {}
    for b in range(n_bins):
        view = _PlainView(ds_cond, bin_to_idxs[b])
        loaders[b] = DataLoader(view, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return loaders


def build_per_bin_plain_loaders(
    gerbil_roots,
    families: Iterable[int],
    *,
    specs_per_file: int,
    batch_size: int,
    num_workers: int,
    test_size: float,
    split_seed: int,
    conditional_factor: str,
    n_bins: int = 3,
) -> Dict[int, dict]:
    """
    One pass to bin items via conditional factor, then return per-bin PLAIN loaders.
    Returns {bin: {"train": loader or None, "test": loader or None, "n_train": int, "n_test": int}}
    """
    (train_fns, test_fns), (train_ids, test_ids), specs_per_file = load_gerbils_multi(
        gerbil_filepath=gerbil_roots,
        specs_per_file=specs_per_file,
        families=families,
        test_size=test_size,
        seed=split_seed,
        check=True,
    )

    # Conditional datasets to read c and decide bin
    ds_tr_cond = bird_data(
        train_fns,
        train_ids,
        specs_per_file=specs_per_file,
        transform=spec_to_tensor,
        conditional=True,
        conditional_factor=conditional_factor,
    )
    ds_te_cond = bird_data(
        test_fns,
        test_ids,
        specs_per_file=specs_per_file,
        transform=spec_to_tensor,
        conditional=True,
        conditional_factor=conditional_factor,
    )

    # Plain datasets for actual evaluation
    ds_tr_plain = bird_data(train_fns, train_ids, specs_per_file=specs_per_file, transform=spec_to_tensor, conditional=False)
    ds_te_plain = bird_data(test_fns, test_ids, specs_per_file=specs_per_file, transform=spec_to_tensor, conditional=False)

    bin_to_idx_tr = gather_bin_indices(ds_tr_cond, n_bins=n_bins)
    bin_to_idx_te = gather_bin_indices(ds_te_cond, n_bins=n_bins)

    pin = bool(torch.cuda.is_available())
    out = {}
    for b in range(n_bins):
        tr_subset = Subset(ds_tr_plain, bin_to_idx_tr[b])
        te_subset = Subset(ds_te_plain, bin_to_idx_te[b])
        ntr, nte = len(tr_subset), len(te_subset)

        if ntr == 0 and nte == 0:
            out[b] = {"train": None, "test": None, "n_train": 0, "n_test": 0}
            continue

        bs_tr = min(batch_size, max(1, ntr))
        bs_te = min(batch_size, max(1, nte))
        tr_loader = DataLoader(tr_subset, batch_size=bs_tr, shuffle=False, num_workers=num_workers, pin_memory=pin) if ntr > 0 else None
        te_loader = DataLoader(te_subset, batch_size=bs_te, shuffle=False, num_workers=num_workers, pin_memory=pin) if nte > 0 else None
        out[b] = {"train": tr_loader, "test": te_loader, "n_train": ntr, "n_test": nte}
    return out


# -----------------------
# Optional metadata helpers
# -----------------------


def _read_locations_for_file(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        locs = f["locations"][:]
    return np.array([x.decode("ASCII") for x in locs], dtype=object)


def build_locations_vector(file_list: List[str]) -> np.ndarray:
    """Concatenate per-file 'locations' datasets to align with dataset iteration order (file-major)."""
    out: List[str] = []
    for p in file_list:
        out.extend(_read_locations_for_file(p))
    return np.array(out, dtype=object)


def loc_bucketize(loc_str_array: np.ndarray) -> np.ndarray:
    """Map raw location strings to coarse buckets (arena / underground)."""
    s = np.char.lower(loc_str_array.astype(str))
    arena_mask = np.logical_or(np.char.find(s, "arena_1") >= 0, np.char.find(s, "arena_2") >= 0)
    return np.where(arena_mask, "arena", "underground")

# Visualize
# 3D

@torch.no_grad()
def plot_decoder_slices_3d(
    model,
    z3_values,
    n_samples_dim=20,
    device=None,
    origin="lower",
    cm="inferno",
    show=False,
    save_prefix=None,
):
    """
    Visualize decoder cross-sections for a 3D latent space.

    For each z3 in `z3_values`, this:
      - builds an n_samples_dim x n_samples_dim grid over (z1, z2) in [0,1]^2
      - fixes z3 = constant
      - decodes all points
      - tiles the decoded spectrograms into a big image
      - optionally saves each slice as PNG

    Args
    ----
    model : QMCLVM
        Trained model with latent_dim = 3.
    z3_values : list/tuple of float
        Values in [0,1] at which to slice along the third latent dimension.
    n_samples_dim : int
        Number of points per axis for z1 and z2 grid.
    device : torch.device or None
        Device for computation; if None, uses model.device.
    origin : {"lower","upper"}
        Passed to imshow; "lower" means (0,0) at bottom-left.
    cm : str
        Matplotlib colormap name.
    show : bool
        Whether to call plt.show() at the end for each slice.
    save_prefix : str or None
        If not None, saves each slice as f"{save_prefix}_z3_{z3:.2f}.png".
    """
    if device is None:
        device = model.device

    model.eval()

    # Latent grid over z1,z2 in [0,1]
    lin = torch.linspace(0.0, 1.0, n_samples_dim)
    Z1, Z2 = torch.meshgrid(lin, lin, indexing="ij")  # [n,n]

    # We'll fill these per-slice
    for z3 in z3_values:
        # Build [K,3] grid: (z1,z2,z3_fixed)
        K = n_samples_dim * n_samples_dim
        z1_flat = Z1.reshape(-1)
        z2_flat = Z2.reshape(-1)
        z3_flat = torch.full_like(z1_flat, float(z3))

        eval_grid = torch.stack([z1_flat, z2_flat, z3_flat], dim=1)  # [K,3]
        eval_grid = eval_grid.to(device=device, dtype=torch.float32)

        # Forward pass through QLVM decoder
        # random=False, mod=False because we're specifying exact z in [0,1]^3
        samples = model(eval_grid, random=False, mod=False)  # [K, C, H, W]

        # Assume single-channel spectrograms: C=1
        if samples.dim() != 4 or samples.shape[1] != 1:
            raise ValueError(f"Expected decoder output [K,1,H,W], got {samples.shape}")

        specs = samples[:, 0].cpu().numpy()  # [K, H, W], in [0,1] after sigmoid
        K, H, W = specs.shape

        # Build big tiled image: (n_samples_dim * H, n_samples_dim * W)
        grid_img = np.zeros((n_samples_dim * H, n_samples_dim * W), dtype=np.float32)

        for i in range(n_samples_dim):
            for j in range(n_samples_dim):
                idx = i * n_samples_dim + j
                patch = specs[idx]  # [H,W]

                # If origin="lower", we flip the vertical indexing so that
                # low z2 is at bottom; adjust if you prefer opposite.
                row = (n_samples_dim - 1 - i) if origin == "lower" else i

                r0 = row * H
                r1 = r0 + H
                c0 = j * W
                c1 = c0 + W

                grid_img[r0:r1, c0:c1] = patch

        # Plot
        plt.figure(figsize=(6, 6))
        plt.imshow(
            grid_img,
            cmap=cm,
            origin=origin,
            aspect="auto",
        )
        #plt.colorbar(label="decoder output (πθ)")
        plt.title(f"Decoder slice at z3 = {z3:.2f}")
        plt.axis("off")

        if save_prefix is not None:
            fn = f"{save_prefix}_z3_{z3:.2f}.png"
            plt.savefig(fn, dpi=150, bbox_inches="tight")
            print(f"[slice] wrote {fn}")

        if show:
            plt.show()
        else:
            plt.close()

import torch
import numpy as np
from tqdm import tqdm

def plot_latent_embedding(emb, labels, out_path, title="Test embeddings"):
    """
    Minimal plotting helper:
      - If latent_dim = 2 → 2D scatter
      - If latent_dim = 3 → 3D scatter
    Colors by label if there is more than one unique label.
    """
    emb = np.asarray(emb)
    labels = np.asarray(labels)

    d = emb.shape[1]
    uniq = np.unique(labels)

    # simple color handling
    if len(uniq) <= 1:
        colors = "C0"
    else:
        cmap = plt.get_cmap("tab10")
        color_map = {u: cmap(i % 10) for i, u in enumerate(uniq)}
        colors = [color_map[l] for l in labels]

    if d == 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(emb[:, 0], emb[:, 1], s=5, c=colors, alpha=0.8)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
        ax.set_title(title)
        ax.set_aspect("equal", "box")

        if len(uniq) > 1:
            handles = []
            for u in uniq:
                handles.append(
                    plt.Line2D([], [], marker="o", linestyle="",
                               color=color_map[u], label=str(u), markersize=6)
                )
            ax.legend(handles=handles, frameon=False, title="label", loc="best")

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    elif d == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(
            emb[:, 0], emb[:, 1], emb[:, 2],
            s=5, c=colors, alpha=0.7
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
        ax.set_zlabel("z₃")
        ax.set_title(title)

        if len(uniq) > 1:
            handles = []
            for u in uniq:
                handles.append(
                    plt.Line2D([], [], marker="o", linestyle="",
                               color=color_map[u], label=str(u), markersize=6)
                )
            ax.legend(handles=handles, frameon=False, title="label", loc="best")

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    else:
        raise ValueError(f"Don't know how to plot latent_dim={d} (expected 2 or 3)")


def compute_nll_and_energy(model, latent_grid, loader, loss_function):
    device = model.device
    latent_grid = latent_grid.to(device)

    all_nll = []
    all_energy = []

    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device).float()        # [B, C, H, W]
            samples = model(latent_grid, random=True, mod=True)

            # per-example NLL (same metric as training)
            nll_batch = loss_function(
                samples,
                data,
                reduce=False,
                importance_weights=[]
            )  # [B]

            # per-example energy
            energy_batch = (data ** 2).sum(dim=(1, 2, 3))  # [B]

            all_nll.append(nll_batch.cpu().numpy())
            all_energy.append(energy_batch.cpu().numpy())

    all_nll = np.concatenate(all_nll, axis=0)
    all_energy = np.concatenate(all_energy, axis=0)
    return all_nll, all_energy


def collect_examples_from_loader(loader, target_indices):
    """
    Collect original spectrograms (no recon) for given global indices.
    Returns: array [Nsel, C, H, W]
    """
    target_indices = np.sort(np.array(target_indices))
    target_set = set(target_indices.tolist())

    collected = []
    global_i = 0

    with torch.no_grad():
        for batch in loader:
            data, _ = batch
            B = data.shape[0]

            for j in range(B):
                if global_i in target_set:
                    collected.append(data[j].cpu().numpy())
                global_i += 1

    if len(collected) == 0:
        return np.zeros((0,))
    return np.stack(collected, axis=0)  # [Nsel, C, H, W]

def plot_montage_100(originals, scores, out_path, title_prefix):
    """
    Plot up to 100 spectrograms in a 10x10 grid.
    `scores` is per-example NLL; we annotate each tile with its value.
    """
    N = originals.shape[0]
    n_show = min(100, N)

    fig, axes = plt.subplots(10, 10, figsize=(12, 12))
    for idx in range(n_show):
        row, col = divmod(idx, 10)
        ax = axes[row, col]

        img = originals[idx]
        # handle CxHxW or 1xHxW
        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]
        elif img.ndim == 3 and img.shape[0] > 1:
            # just take first channel
            img = img[0]

        ax.imshow(img, cmap="inferno", origin="lower", aspect="auto")
        ax.axis("off")
        ax.text(
            0.01, 0.99,
            f"{scores[idx]:.1f}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=6, color="white"
        )

    fig.suptitle(f"{title_prefix} — top {n_show}", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# @torch.no_grad()
# def compute_3d_embeddings(model, base_sequence, data_loader, device):
#     """
#     Compute 3D latent embeddings for each spectrogram in `data_loader`.
#
#     Returns:
#         z_embeds: (N, 3) numpy array of posterior-mean latent coords
#         labels:  (N,) numpy array of labels (if provided by dataset)
#     """
#     model.eval()
#
#     # base_sequence: [K, 3]
#     base_seq = base_sequence.to(device).to(torch.float32)  # ensure on same device/dtype
#
#     # Precompute decoder output once: [K, C, H, W]
#     samples = model(base_seq, random=False, mod=True)
#
#     all_z = []
#     all_labels = []
#
#     for batch in tqdm(data_loader, desc="Embedding"):
#         # data_loader yields (spec, label)
#         if len(batch) == 2:
#             x, labels = batch
#         else:
#             # Fallback if dataset only returns specs
#             x = batch[0]
#             labels = None
#
#         x = x.to(device)
#
#         # log p(x | z_j) for all z_j
#         log_px_z = binary_lp(samples, x)  # shape: [B, K]
#
#         # posterior over z: softmax along K
#         w = torch.softmax(log_px_z, dim=1)  # [B, K]
#
#         # posterior mean z: [B, K] @ [K, 3] = [B, 3]
#         z_mean = w @ base_seq  # [B, 3]
#
#         all_z.append(z_mean.cpu().numpy())
#         if labels is not None:
#             all_labels.append(labels.cpu().numpy())
#
#     z_embeds = np.concatenate(all_z, axis=0)
#     if all_labels:
#         labels = np.concatenate(all_labels, axis=0)
#     else:
#         labels = None
#
#     return z_embeds, labels

# R^2
def compute_batch_r2(x, x_recon, var_eps=1e-6):
    """
    x, x_recon: [B, C, H, W]
    Returns:
        r2:  [B] numpy array of R^2 values
        var: [B] numpy array of total variance per example (SS_tot)
    """
    B = x.shape[0]
    # flatten per example
    x_flat = x.view(B, -1)
    xr_flat = x_recon.view(B, -1)

    # residual sum of squares
    ss_res = ((x_flat - xr_flat) ** 2).sum(dim=1)          # [B]

    # total sum of squares (variance of x around its own mean)
    x_mean = x_flat.mean(dim=1, keepdim=True)              # [B, 1]
    ss_tot = ((x_flat - x_mean) ** 2).sum(dim=1)           # [B]

    # R^2; set to NaN for near-zero variance
    r2 = torch.empty_like(ss_tot)
    valid = ss_tot > var_eps

    r2[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
    r2[~valid] = torch.nan

    return r2.cpu().numpy(), ss_tot.cpu().numpy()


def compute_r2_over_loader(model, latent_grid, loader, log_likelihood, n_samples=5, c=[]):
    """
    For each example in `loader`, compute posterior-mean reconstruction
    and its R^2 w.r.t. the true spectrogram.

    Uses model.round_trip(..., recon_type='posterior').

     Returns:
      r2_all:   [N_examples] R^2 (NaN for near-constant / silent examples)
      var_all:  [N_examples] total variance (SS_tot) per example
    """
    device = model.device
    grid = latent_grid.to(device)

    r2_list = []
    var_list = []

    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device).float()  # [B,C,H,W]

            recon = model.round_trip(
                grid=grid,
                data=data,
                log_likelihood=log_likelihood,  # binary_lp
                recon_type='posterior',
                n_samples=n_samples,
                c=c,
            )  # [B,C,H,W]

            r2_batch, var_batch = compute_batch_r2(data, recon)
            r2_list.append(r2_batch)
            var_list.append(var_batch)

    r2_all = np.concatenate(r2_list, axis=0)
    var_all = np.concatenate(var_list, axis=0)
    return r2_all, var_all

def collect_recons_from_loader(model, latent_grid, loader, target_indices, log_likelihood,
                               n_samples=5, c=[]):
    """
    For the selected global indices (target_indices),
    compute posterior-mean reconstructions.

    Returns:
        recons: array [Nsel, C, H, W]  of reconstructions
    """
    device = model.device
    grid = latent_grid.to(device)

    target_indices = np.sort(np.array(target_indices))
    target_set = set(target_indices.tolist())

    collected = []
    global_i = 0

    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device).float()
            B = data.shape[0]

            # compute reconstructions for the whole batch
            recon = model.round_trip(
                grid=grid,
                data=data,
                log_likelihood=log_likelihood,
                recon_type="posterior",
                n_samples=n_samples,
                c=c,
            )  # [B,C,H,W]

            # pick only those whose global index matches our target indices
            for j in range(B):
                if global_i in target_set:
                    collected.append(recon[j].cpu().numpy())
                global_i += 1

    if len(collected) == 0:
        return np.zeros((0,))
    return np.stack(collected, axis=0)




def plot_montage_pairs(true_specs, recon_specs, scores, out_path, title_prefix):
    """
    true_specs:  [N, C, H, W]
    recon_specs: [N, C, H, W]
    scores:      [N] (e.g. R^2)

    Layout: 10 rows, 10 pairs per row.
    Each pair is: [ real | recon | gap ] horizontally.
    """
    N = min(100, true_specs.shape[0])
    n_rows = 10
    n_pairs_per_row = 10
    n_cols = n_pairs_per_row * 3  # real, recon, gap

    # Make real/recon columns wide and gap columns narrow
    width_ratios = []
    for _ in range(n_pairs_per_row):
        width_ratios.extend([1.0, 1.0, 0.25])  # real, recon, gap

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(22, 10),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.05, "hspace": 0.05},
    )

    # If axes comes back as 1D in some weird edge case, make sure it's 2D
    if axes.ndim == 1:
        axes = axes[None, :]

    for k in range(N):
        row = k // n_pairs_per_row
        pair_idx = k % n_pairs_per_row

        col_real  = 3 * pair_idx      # real image column
        col_recon = col_real + 1      # recon image column
        col_gap   = col_real + 2      # gap column (left unused)

        # --- REAL ---
        ax_real = axes[row, col_real]
        img_real = true_specs[k, 0] if true_specs[k].ndim == 3 else true_specs[k]
        ax_real.imshow(img_real, cmap="inferno", origin="lower", aspect="auto")
        ax_real.axis("off")

        # small "real" label
        ax_real.text(
            0.02, 0.02, "real",
            transform=ax_real.transAxes,
            ha="left", va="bottom",
            fontsize=6, color="white",
            bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"),
        )

        # --- RECON ---
        ax_recon = axes[row, col_recon]
        img_rec = recon_specs[k, 0] if recon_specs[k].ndim == 3 else recon_specs[k]
        ax_recon.imshow(img_rec, cmap="inferno", origin="lower", aspect="auto")
        ax_recon.axis("off")

        # small "recon" label
        ax_recon.text(
            0.02, 0.02, "recon",
            transform=ax_recon.transAxes,
            ha="left", va="bottom",
            fontsize=6, color="white",
            bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"),
        )

        # R^2 (or whatever score) in the top-left of recon
        ax_recon.text(
            0.02, 0.98, f"{scores[k]:.2f}",
            transform=ax_recon.transAxes,
            ha="left", va="top",
            fontsize=6, color="white",
            bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"),
        )

        # --- GAP column ---
        ax_gap = axes[row, col_gap]
        ax_gap.axis("off")  # empty column -> visual horizontal spacing

    fig.suptitle(f"{title_prefix}", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

## stuff for reading hdf5 keys
def _loc_bucketize(loc_str_array: np.ndarray) -> np.ndarray:
    s = np.char.lower(loc_str_array.astype(str))
    arena_mask = np.logical_or(np.char.find(s, "arena_1") >= 0,
                               np.char.find(s, "arena_2") >= 0)
    return np.where(arena_mask, "arena", "underground")

# plot
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