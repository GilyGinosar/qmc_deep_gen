"""
Thin evaluation script for a single unconditional QMCLVM trained on all families.
Outputs:
  - test losses (npy) + eval metadata (json)
  - train vs test evidence plot
  - decoder grid visualization
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from models.qmc_base import QMCLVM
from models.sampling import gen_fib_basis
from models.utils import get_decoder_arch
from plotting.visualize import model_grid_plot, qmc_train_plot
from train.losses import binary_evidence
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
CKPT_DIR = r"D:\data\Figs_uncond_all\model_checkpoints\uncond_all_20251015_130934"
OUT_DIR_OVERRIDE = None  # leave None to write under CKPT_DIR/eval_<run_id>

BATCH_SIZE = 64
NUM_WORKERS = 0
SPECS_PER_FILE = 100
TEST_SIZE = 0.20
SPLIT_SEED = 92
LATENT_DIM = 2
M_FIB = 15


# -----------------------
# Model helper
# -----------------------
def rebuild_unconditional_model(device: torch.device) -> QMCLVM:
    decoder = get_decoder_arch(dataset_name="gerbil_ava", latent_dim=LATENT_DIM, arch="qmc")
    return QMCLVM(latent_dim=LATENT_DIM, device=device, decoder=decoder)


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
    train_loader, test_loader, (train_fns, test_fns) = eu.build_plain_loaders(
        DATA_ROOT,
        FAMILIES,
        specs_per_file=SPECS_PER_FILE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        test_size=TEST_SIZE,
        split_seed=SPLIT_SEED,
    )
    print("[data] train batches:", len(train_loader), "test batches:", len(test_loader))

    # 2) Model + checkpoint
    ckpt_path = eu.latest_checkpoint(CKPT_DIR)
    print("[ckpt] loading", ckpt_path)
    model = rebuild_unconditional_model(device)
    train_losses, epoch = eu.load_model_weights(model, ckpt_path, device)
    latent_grid = gen_fib_basis(m=M_FIB)

    run_id = Path(ckpt_path).stem
    out_dir = OUT_DIR_OVERRIDE or os.path.join(CKPT_DIR, f"eval_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    # 3) Test loss
    with torch.no_grad():
        test_losses = test_epoch(
            model,
            test_loader,
            latent_grid.to(device),
            binary_evidence,
            conditional=False,
        )

    np.save(os.path.join(out_dir, "test_losses.npy"), np.asarray(test_losses, np.float32))
    with open(os.path.join(out_dir, "eval_meta.json"), "w") as f:
        json.dump(
            {
                "ckpt": os.path.abspath(ckpt_path),
                "epoch": epoch,
                "n_test_batches": len(test_loader),
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

    model_grid_plot(
        model,
        n_samples_dim=20,
        origin="lower",
        cm="inferno",
        show=False,
        fn=os.path.join(out_dir, "decoder_grid_uncond.png"),
    )

    print("[done]")


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
