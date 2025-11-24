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
    blob = torch.load(ckpt_path, map_location="cpu")
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
