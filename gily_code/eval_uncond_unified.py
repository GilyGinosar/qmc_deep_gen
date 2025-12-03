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
from models.sampling import gen_fib_basis, gen_korobov_basis
from models.utils import get_decoder_arch
from plotting.visualize import qmc_train_plot #, model_grid_plot
from plotting.visualize_3d import model_grid_plot

from train.losses import binary_evidence, binary_lp
from train.train import test_epoch

from gily_code import eval_utils as eu

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch.nn.functional as F
import matplotlib.pyplot as plt




# -----------------------
# CONFIG
# -----------------------
DATA_ROOT = {
    1: [r"D:\Data\Data_vae\235", r"D:\Data\Data_vae\237"],
   # 2: [r"D:\Data\113", r"D:\Data\114", r"D:\Data\115", r"D:\Data\116"],
}
FAMILIES = [1]
CKPT_DIR = r"D:\data\model_checkpoints\uncond_all_20251130_210932"
#CKPT_DIR = r"D:\data\Figs_uncond_all\model_checkpoints"

OUT_DIR_OVERRIDE = None  # leave None to write under CKPT_DIR/eval_<run_id>

BATCH_SIZE = 64
NUM_WORKERS = 0
SPECS_PER_FILE = 100
TEST_SIZE = 0.20
SPLIT_SEED = 92
LATENT_DIM = 3
M_FIB = 15
# Korobov 3D
N_LATENT_POINTS = 1021  # or 2039, 4093
a_korobov = 76

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
    latent_grid = gen_korobov_basis(a=a_korobov,num_dims=LATENT_DIM,num_points=N_LATENT_POINTS) #gen_fib_basis(m=M_FIB)

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

    # # 3b) Per-example NLL on test set
    # per_example_nll, per_example_energy = eu.compute_nll_and_energy(
    #     model,
    #     latent_grid,
    #     test_loader,
    #     binary_evidence
    # )
    #
    # print("Energy stats: min", per_example_energy.min(),
    #       "max", per_example_energy.max(),
    #       "mean", per_example_energy.mean(),
    #       "median", np.median(per_example_energy))
    #
    # # NON TINY CALLS
    # energy_max = per_example_energy.max()
    # energy_median = np.median(per_example_energy)
    # # choose 10% or 30% of that
    # frac = 0.3  # 0.1 for 10%, 0.3 for 30%
    # thr = frac * energy_median
    #
    # non_tiny_mask = per_example_energy >= thr
    # non_tiny_indices = np.where(non_tiny_mask)[0]
    #
    # print("Total examples:", len(per_example_energy))
    # print("Non-tiny examples:", len(non_tiny_indices))
    #
    # nll_non_tiny = per_example_nll[non_tiny_mask]
    # idx_non_tiny = np.where(non_tiny_mask)[0]  # mapping back to global indices
    #
    # sorted_non_tiny = np.argsort(nll_non_tiny)
    #
    # best_100_idx = idx_non_tiny[sorted_non_tiny[:100]]
    # worst_100_idx = idx_non_tiny[sorted_non_tiny[-100:]]
    #
    # # SILENT CALLS
    # # e.g. anything with almost no energy is "silent"
    # # thr = np.quantile(per_example_energy, 0.1)  # or a fixed small value
    # # non_silent_mask = per_example_energy > thr
    # # non_silent_indices = np.where(non_silent_mask)[0]
    # #
    # # print("Total examples:", len(per_example_nll))
    # # print("Non-silent examples:", len(non_silent_indices))
    # #
    # # nll_non_silent = per_example_nll[non_silent_mask]
    # # idx_non_silent = np.where(non_silent_mask)[0]
    # #
    # # sorted_non_silent = np.argsort(nll_non_silent)  # indices within the non-silent subset
    # #
    # # best_non_silent_100 = idx_non_silent[sorted_non_silent[:100]]
    # # worst_non_silent_100 = idx_non_silent[sorted_non_silent[-100:]]


    # 3b) Per-example R^2 on test set (posterior-mean reconstructions)
    r2_test, var_test = eu.compute_r2_over_loader(
        model=model,
        latent_grid=latent_grid,
        loader=test_loader,
        log_likelihood=binary_lp,  # important: binary_lp, not binary_evidence
        n_samples=5,
        c=[]                        # unconditional
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
    sorted_valid = np.argsort(-r2_valid)    # descending: best first
    best_100_valid  = sorted_valid[:100]
    worst_100_valid = sorted_valid[-100:]

    best_100_idx  = valid_indices[best_100_valid]
    worst_100_idx = valid_indices[worst_100_valid]

    np.save(os.path.join(out_dir, "best_100_r2_idx.npy"), best_100_idx)
    np.save(os.path.join(out_dir, "worst_100_r2_idx.npy"), worst_100_idx)

    # collect originals (reuse your existing helpers)
    orig_best = eu.collect_examples_from_loader(test_loader, best_100_idx)
    orig_worst = eu.collect_examples_from_loader(test_loader, worst_100_idx)

    # collect reconstructions for same indices
    recon_best = eu.collect_recons_from_loader(
        model, latent_grid, test_loader,
        best_100_idx, binary_lp, n_samples=5, c=[]
    )

    recon_worst = eu.collect_recons_from_loader(
        model, latent_grid, test_loader,
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

    # eu.plot_montage_100(
    #     originals=orig_best,
    #     scores=r2_test[best_100_idx],
    #     out_path=os.path.join(out_dir, "best_100_examples_r2.png"),
    #     title_prefix="Highest R² (best recon, non-silent)"
    # )
    #
    # eu.plot_montage_100(
    #     originals=orig_worst,
    #     scores=r2_test[worst_100_idx],
    #     out_path=os.path.join(out_dir, "worst_100_examples_r2.png"),
    #     title_prefix="Lowest R² (worst recon, non-silent)"
    # )

    #########
    # orig_best = eu.collect_examples_from_loader(test_loader, best_100_idx)
    # orig_worst = eu.collect_examples_from_loader(test_loader, worst_100_idx)
    #
    #
    # # best 100 (lowest NLL)
    # eu.plot_montage_100(
    #     originals=orig_best,
    #     scores=per_example_nll[best_100_idx],
    #     out_path=os.path.join(out_dir, "best_100_examples_nll.png"),
    #     title_prefix="Lowest NLL (best modeled)"
    # )
    #
    # # worst 100 (highest NLL)
    # eu.plot_montage_100(
    #     originals=orig_worst,
    #     scores=per_example_nll[worst_100_idx],
    #     out_path=os.path.join(out_dir, "worst_100_examples_nll.png"),
    #     title_prefix="Highest NLL (worst modeled)"
    # )

    print("[out] wrote best_100_examples_nll.png and worst_100_examples_nll.png")

    # 4) Embed data
    with torch.no_grad(): # output: latents and labels
        emb_te, lab_te = model.embed_data(
            latent_grid.to(device),  # shape [S, 2] or [S, 3]
            test_loader,
            binary_lp, #log likelihood used
            embed_type="rqmc",  # or "posterior", or "argmax"
            n_samples=5,
            c=[]  # unconditional
        )

    # save embeddings if you want to reuse later
    np.save(os.path.join(out_dir, "emb_te.npy"), emb_te)
    np.save(os.path.join(out_dir, "lab_te.npy"), lab_te)

    # plot
    eu.plot_latent_embedding(
        emb=emb_te,
        labels=lab_te,
        out_path=os.path.join(out_dir, "latent_embedding.png"),
        title=f"Unconditional QLVM (latent_dim={LATENT_DIM})"
    )
    print("[out] wrote latent_embedding.png")

    # 4) Plots
    qmc_train_plot(
        train_losses,
        test_losses,
        save_fn=os.path.join(out_dir, "train_vs_test.png"),
        show=False,
    )
    print("[out] wrote train plot")

    # for 2D
    #model_grid_plot(
    #     model,
    #     n_samples_dim=20,
    #     origin="lower",
    #     cm="inferno",
    #     show=False,
    #     fn=os.path.join(out_dir, "decoder_grid_uncond.png"),
    # )

    # Gily's visualization
    #z3_values = np.linspace(0,1,10)
    # eu.plot_decoder_slices_3d(
    #     model,
    #     z3_values=z3_values,
    #     n_samples_dim=20,
    #     device=device,
    #     save_prefix=os.path.join(out_dir, "decoder_slice"),
    # )

    # Miles' visualization
    model_grid_plot(model, n_samples_dim=20, fn='', show=True, origin=None, cm='grey', model_type='qmc')
    print("[done]")


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
