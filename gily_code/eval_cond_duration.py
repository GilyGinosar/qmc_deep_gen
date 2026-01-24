"""
Evaluation script for a single *conditional* QMCLVM trained on duration/length.

Assumes:
  - Training used COND=True, CF="length"
  - LATENT_DIM = 2 (Fibonacci lattice)
Outputs:
  - test losses (npy) + eval metadata (json)
  - train vs test evidence plot
  - decoder grid visualizations for several duration values
  - optional latent embeddings for a few duration slices
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
    1: [r"D:\Data\Data_vae\235", r"D:\Data\Data_vae\237"],
}
FAMILIES = [1]

# Point this to your conditional run directory
CKPT_DIR = r"D:\data\model_checkpoints\cond_fm"

OUT_DIR_OVERRIDE = None  # leave None to write under CKPT_DIR/eval_<run_id>

BATCH_SIZE = 64          # just affects eval speed
NUM_WORKERS = 0
SPECS_PER_FILE = 100
TEST_SIZE = 0.20
SPLIT_SEED = 92

LATENT_DIM = 2           # conditional duration run is 2D
M_FIB = 15               # Fibonacci lattice density

# Conditional factor name in bird_data.py (matches training)
COND_FACTOR = "fm_scalar"   # our normalized scalar duration (0..1)


# -----------------------
# Model helper
# -----------------------
def rebuild_conditional_model(device: torch.device) -> QMCLVM:
    """
    Rebuild the conditional QMCLVM exactly as in training:
      - 2D latent
      - cond_dim = 1 (scalar duration)
      - 'conditional_qmc' decoder
    """
    decoder = get_decoder_arch(
        dataset_name="gerbil_ava",
        latent_dim=LATENT_DIM,
        arch="conditional_qmc",
        cond_dim=1,          # scalar condition
    )
    return QMCLVM(latent_dim=LATENT_DIM, device=device, decoder=decoder)


class CondWrapped(nn.Module):
    """
    Tiny adapter: fix c to a given scalar (already [1,1] tensor),
    so model_grid_plot can call forward(z, mod, random) without knowing about c.
    """

    def __init__(self, base_model: nn.Module, c_vec: torch.Tensor):
        super().__init__()
        self.base = base_model
        self.c = c_vec

    def forward(self, z, mod=False, random=False):
        return self.base(z, random=random, mod=mod, c=self.c)

    def __getattr__(self, name):
        # delegate everything else to base_model (device, decoder, etc.)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)


# -----------------------
# Main
# -----------------------
def main():
    # ----- seeding & device -----
    torch.manual_seed(92)
    np.random.seed(92)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(92)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ----- 1) Data: conditional + plain loaders -----
    (
        train_loader_cond,
        test_loader_cond,
        train_loader_plain,
        test_loader_plain,
        (train_fns, test_fns),
    ) = eu.build_conditional_loaders(
        gerbil_roots=DATA_ROOT,
        families=FAMILIES,
        specs_per_file=SPECS_PER_FILE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        test_size=TEST_SIZE,
        split_seed=SPLIT_SEED,
        conditional_factor=COND_FACTOR,
    )

    print(
        "[data] batches:",
        len(train_loader_cond),
        len(test_loader_cond),
    )

    # ----- 2) Model + checkpoint -----
    ckpt_path = eu.latest_checkpoint(CKPT_DIR)
    print("[ckpt] loading:", ckpt_path)

    model = rebuild_conditional_model(device)
    train_losses, epoch = eu.load_model_weights(model, ckpt_path, device)

    latent_grid = gen_fib_basis(m=M_FIB).to(device).float()  # [S, 2]

    run_id = Path(ckpt_path).stem
    out_dir = OUT_DIR_OVERRIDE or os.path.join(CKPT_DIR, f"eval_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] eval dir: {out_dir}")

    # ----- 3) Test loss (same loss as training) -----
    with torch.no_grad():
        test_losses = test_epoch(
            model,
            test_loader_cond,      # (spec, c, label)
            latent_grid,
            binary_evidence,
            conditional=True,
        )

    np.save(os.path.join(out_dir, "test_losses.npy"), np.asarray(test_losses, np.float32))
    with open(os.path.join(out_dir, "eval_meta.json"), "w") as f:
        json.dump(
            {
                "ckpt": os.path.abspath(ckpt_path),
                "epoch": epoch,
                "n_test_batches": len(test_loader_cond),
                "device": str(device),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "train_files": train_fns,
                "test_files": test_fns,
                "latent_dim": LATENT_DIM,
                "cond_factor": COND_FACTOR,
            },
            f,
            indent=2,
        )
    print(f"[out] wrote eval artifacts to {out_dir}")

    # ----- 4) Train vs test plot -----
    qmc_train_plot(
        train_losses,
        test_losses,
        save_fn=os.path.join(out_dir, "train_vs_test.png"),
        show=False,
    )
    print("[out] wrote train_vs_test plot")

    # ----- 5) Decoder grid for several duration values -----
    # visualize a few c values in [0,1] (normalized duration).
    # c_values = np.linspace(0.0, 1.0, 5)
    #
    # for c_val in c_values:
    #     c_vec = torch.tensor([[float(c_val)]], device=device, dtype=torch.float32)
    #     print(f"[viz] duration c={c_val:.2f}")
    #
    #     cond_model = CondWrapped(model, c_vec)
    #
    #     with torch.no_grad():
    #         model_grid_plot(
    #             cond_model,
    #             n_samples_dim=20,
    #             fn=os.path.join(out_dir, f"decoder_grid_dur_{c_val:.2f}.png"),
    #             show=False,
    #             origin="lower",
    #             cm="inferno",
    #             model_type="qmc",
    #         )

    # 3b) Per-example R^2 on test set (posterior-mean reconstructions)
    r2_test, var_test = eu.compute_r2_over_loader(
        model=model,
        latent_grid=latent_grid,
        loader=test_loader_plain,
        log_likelihood=binary_lp,  # important: binary_lp, not binary_evidence
        n_samples=5,
        c=[]  # unconditional
    )

    # keep only examples with non-trivial variance (avoid silent / flat specs)
    thr_var = np.quantile(var_test, 0.4)  # drop the lowest XX% variance examples
    valid_mask = (~np.isnan(r2_test)) & (var_test >= thr_var)

    valid_indices = np.where(valid_mask)[0]
    r2_valid = r2_test[valid_mask]

    print("R^2 stats on test set (valid, non-silent examples):")
    print("  N valid =", len(r2_valid), "out of", len(r2_test))
    print("  min     =", float(np.nanmin(r2_valid)))
    print("  max     =", float(np.nanmax(r2_valid)))
    print("  mean    =", float(np.nanmean(r2_valid)))
    print("  median  =", float(np.nanmedian(r2_valid)))

    np.save(os.path.join(out_dir, "r2_test.npy"), r2_test)
    np.save(os.path.join(out_dir, "var_test.npy"), var_test)

    # rank among valid examples only
    sorted_valid = np.argsort(-r2_valid)  # descending: best first
    best_100_valid = sorted_valid[:100]
    worst_100_valid = sorted_valid[-100:]

    best_100_idx = valid_indices[best_100_valid]
    worst_100_idx = valid_indices[worst_100_valid]

    np.save(os.path.join(out_dir, "best_100_r2_idx.npy"), best_100_idx)
    np.save(os.path.join(out_dir, "worst_100_r2_idx.npy"), worst_100_idx)

    # collect originals (reuse your existing helpers)
    orig_best = eu.collect_examples_from_loader(test_loader_plain, best_100_idx)
    orig_worst = eu.collect_examples_from_loader(test_loader_plain, worst_100_idx)

    # collect reconstructions for same indices
    recon_best = eu.collect_recons_from_loader(
        model, latent_grid, test_loader_plain,
        best_100_idx, binary_lp, n_samples=5, c=[]
    )

    recon_worst = eu.collect_recons_from_loader(
        model, latent_grid, test_loader_plain,
        worst_100_idx, binary_lp, n_samples=5, c=[]
    )

    # paired montage (true + recon)
    eu.plot_montage_pairs(
        true_specs=orig_best,
        recon_specs=recon_best,
        scores=r2_test[best_100_idx],
        out_path=os.path.join(out_dir, "best_100_pairs_r2.png"),
        title_prefix="Best 100 R² examples"
    )

    eu.plot_montage_pairs(
        true_specs=orig_worst,
        recon_specs=recon_worst,
        scores=r2_test[worst_100_idx],
        out_path=os.path.join(out_dir, "worst_100_pairs_r2.png"),
        title_prefix="Worst 100 R² examples"
    )

    # ----- 6) Optional: embed test data for a few duration slices -----
    # This uses the *plain* loader (spec, label) and passes c explicitly.
    # Latents live in [0,1]^2; scatter for a quick sanity check.
    def simple_embed_and_plot(c_val: float):
        c_vec = torch.tensor([[float(c_val)]], device=device, dtype=torch.float32)
        print(f"[embed] duration c={c_val:.2f}")

        latents, labels = model.embed_data(
            grid=latent_grid.to(device),
            loader=test_loader_plain,   # (spec, label)
            log_likelihood=binary_lp,   # per-example log p(x|z)
            embed_type="rqmc",
            n_samples=5,
            c=c_vec,
        )

        # latents: [N, 2] numpy
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        ax.scatter(latents[:, 0], latents[:, 1], s=4, alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("latent dim 1")
        ax.set_ylabel("latent dim 2")
        ax.set_title(f"Embeddings, duration c={c_val:.2f}")
        ax.set_aspect("equal", "box")
        plt.tight_layout()
        out_fn = os.path.join(out_dir, f"embeddings_dur_{c_val:.2f}.png")
        plt.savefig(out_fn, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[embed] wrote {out_fn}")

    # Pick a small subset of durations to embed (start, middle, end)
    for c_val in [0.0, 0.5, 1.0]:
        simple_embed_and_plot(c_val)

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
