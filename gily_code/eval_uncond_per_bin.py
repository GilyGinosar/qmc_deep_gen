"""
Evaluate three separate unconditional QMCLVMs (one per bin).
Assumes checkpoints live under CKPT_ROOT/bin0, bin1, bin2.
Outputs per bin:
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
    1: [r"D:\Data\237"],
}
FAMILIES = [1]
CKPT_ROOT = r"D:\data\model_checkpoints"  # expects bin0/, bin1/, bin2/ inside
OUT_ROOT_OVERRIDE = None  # set to override output root; default is per-bin under CKPT_ROOT

BATCH_SIZE = 64
NUM_WORKERS = 0
SPECS_PER_FILE = 100
TEST_SIZE = 0.20
SPLIT_SEED = 92
LATENT_DIM = 2
M_FIB = 15
COND_FACTOR_FOR_BINNING = "rule3_bands"  # used only to decide bin ids via c argmax
N_BINS = 3


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

    # 1) Per-bin plain loaders (binning via conditional factor)
    bin_loaders = eu.build_per_bin_plain_loaders(
        DATA_ROOT,
        FAMILIES,
        specs_per_file=SPECS_PER_FILE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        test_size=TEST_SIZE,
        split_seed=SPLIT_SEED,
        conditional_factor=COND_FACTOR_FOR_BINNING,
        n_bins=N_BINS,
    )

    latent_grid = gen_fib_basis(m=M_FIB)

    for b in range(N_BINS):
        entry = bin_loaders[b]
        ntr, nte = entry["n_train"], entry["n_test"]
        if ntr == 0 and nte == 0:
            print(f"[bin {b}] empty; skipping")
            continue

        print(f"\n=== Bin {b} ===")
        print(f"[data] n_train={ntr} n_test={nte}")

        bin_dir = os.path.join(CKPT_ROOT, f"bin{b}")
        ckpt_path = eu.latest_checkpoint(bin_dir)
        print("[ckpt] loading", ckpt_path)

        model = rebuild_unconditional_model(device)
        train_losses, epoch = eu.load_model_weights(model, ckpt_path, device)

        # choose out dir
        out_root = OUT_ROOT_OVERRIDE or bin_dir
        out_dir = os.path.join(out_root, f"eval_bin{b}_epoch{epoch if epoch is not None else 'NA'}")
        os.makedirs(out_dir, exist_ok=True)

        # test loss
        with torch.no_grad():
            test_losses = test_epoch(
                model,
                entry["test"],
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
                    "n_test_batches": len(entry["test"]) if entry["test"] else 0,
                    "device": str(device),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "n_train_items": ntr,
                    "n_test_items": nte,
                },
                f,
                indent=2,
            )
        print(f"[out] wrote artifacts to {out_dir}")

        # plots
        qmc_train_plot(
            train_losses,
            test_losses,
            save_fn=os.path.join(out_dir, f"train_vs_test_bin{b}.png"),
            show=False,
        )
        model_grid_plot(
            model,
            n_samples_dim=20,
            origin="lower",
            cm="inferno",
            show=False,
            fn=os.path.join(out_dir, f"decoder_grid_bin{b}.png"),
        )

    print("\n[done]")


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
