"""
Thin conditional evaluation script.
Uses a single conditional QMCLVM checkpoint (one model, three bins via one-hot c).
Outputs:
  - test losses (npy) + eval metadata (json)
  - train vs test evidence plot
  - decoder grids for each bin (one-hot wrapper)

Adjust the CONFIG section and run:
  python -m gily_code.eval_conditional
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from models.qmc_base import QMCLVM
from models.sampling import gen_fib_basis
from models.utils import get_decoder_arch
from plotting.visualize import model_grid_plot, qmc_train_plot
from train.losses import binary_evidence, binary_lp
from train.train import test_epoch

from gily_code import eval_utils as eu


# -----------------------
# CONFIG
# -----------------------
DATA_ROOT = {
    1: [r"D:\Data\235", r"D:\Data\237"],
    2: [r"D:\Data\113", r"D:\Data\114", r"D:\Data\115", r"D:\Data\116"],
}
FAMILIES = [1, 2]
CKPT_DIR = r"D:\Data\model_checkpoints"  # directory containing final_*.pt / ckpt_*.pt
OUT_DIR_OVERRIDE = None  # set to override output dir; otherwise inside CKPT_DIR/eval_<run_id>

BATCH_SIZE = 64
NUM_WORKERS = 0  # set >0 if Linux; 0 is Windows-safe
SPECS_PER_FILE = 100
TEST_SIZE = 0.20
SPLIT_SEED = 92
LATENT_DIM = 2
M_FIB = 15
COND_FACTOR = "rule3_bands"
COND_DIM = 3


# -----------------------
# Model helpers
# -----------------------
def rebuild_conditional_model(device: torch.device) -> QMCLVM:
    decoder = get_decoder_arch(
        dataset_name="gerbil_ava",
        latent_dim=LATENT_DIM,
        arch="conditional_qmc",
        cond_dim=COND_DIM,
    )
    return QMCLVM(latent_dim=LATENT_DIM, device=device, decoder=decoder)


class _CondWrapper(nn.Module):
    """Wrap decoder to fix a one-hot condition for visualization."""

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


# -----------------------
# Main
# -----------------------
def main():
    torch.manual_seed(92)
    np.random.seed(92)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(92)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # 1) Data
    train_loader_c, test_loader_c, _, _, (train_fns, test_fns) = eu.build_conditional_loaders(
        DATA_ROOT,
        FAMILIES,
        specs_per_file=SPECS_PER_FILE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        test_size=TEST_SIZE,
        split_seed=SPLIT_SEED,
        conditional_factor=COND_FACTOR,
    )

    # Also build per-bin (plain) loaders for potential embeddings / sanity checks
    train_bin_loaders = eu.build_bin_loaders_from_cond(
        train_loader_c.dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        n_bins=COND_DIM,
    )
    test_bin_loaders = eu.build_bin_loaders_from_cond(
        test_loader_c.dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        n_bins=COND_DIM,
    )
    print("[data] train batches:", len(train_loader_c), "test batches:", len(test_loader_c))

    # 2) Model + checkpoint
    ckpt_path = eu.latest_checkpoint(CKPT_DIR)
    print("[ckpt] loading", ckpt_path)
    model = rebuild_conditional_model(device)
    train_losses, epoch = eu.load_model_weights(model, ckpt_path, device)
    latent_grid = gen_fib_basis(m=M_FIB)

    run_id = Path(ckpt_path).stem
    out_dir = OUT_DIR_OVERRIDE or os.path.join(CKPT_DIR, f"eval_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    # 3) Test loss
    with torch.no_grad():
        test_losses = test_epoch(
            model,
            test_loader_c,
            latent_grid.to(device),
            binary_evidence,
            conditional=True,
        )

    np.save(os.path.join(out_dir, "test_losses.npy"), np.asarray(test_losses, np.float32))
    with open(os.path.join(out_dir, "eval_meta.json"), "w") as f:
        json.dump(
            {
                "ckpt": os.path.abspath(ckpt_path),
                "epoch": epoch,
                "n_test_batches": len(test_loader_c),
                "device": str(device),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "train_files": train_fns,
                "test_files": test_fns,
            },
            f,
            indent=2,
        )
    print(f"[out] wrote artifacts to {out_dir}")

    # 4) Plots
    qmc_train_plot(
        train_losses,
        test_losses,
        save_fn=os.path.join(out_dir, "train_vs_test.png"),
        show=False,
    )

    # Per-bin decoder grids (wrap c one-hot)
    onehots = {
        0: torch.tensor([[1.0, 0.0, 0.0]], device=device),
        1: torch.tensor([[0.0, 1.0, 0.0]], device=device),
        2: torch.tensor([[0.0, 0.0, 1.0]], device=device),
    }
    for b in range(COND_DIM):
        wrapped = _CondWrapper(model, onehots[b])
        model_grid_plot(
            wrapped,
            n_samples_dim=20,
            origin="lower",
            cm="inferno",
            show=False,
            fn=os.path.join(out_dir, f"decoder_grid_bin{b}.png"),
        )

    # Optional: uncomment to compute embeddings by bin
    # with torch.no_grad():
    #     emb_te, lab_te = model.embed_data(
    #         latent_grid.to(device),
    #         test_bin_loaders[0],
    #         binary_lp,
    #         embed_type="rqmc",
    #         n_samples=5,
    #         c=onehots[0],
    #     )

    print("[done]")


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
