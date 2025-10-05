# eval_qmc.py
import os, glob, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path  # add near your imports

# ==== project imports (same stack you used for training) ====
from vocalizations.qmc_deep_gen.data.bird_data import load_gerbils, bird_data
from vocalizations.qmc_deep_gen.models.sampling import gen_fib_basis  # latent grid for 2D
from vocalizations.qmc_deep_gen.models.utils import get_decoder_arch
from vocalizations.qmc_deep_gen.models.qmc_base import QMCLVM
from vocalizations.qmc_deep_gen.train.losses import binary_evidence, binary_lp
from vocalizations.qmc_deep_gen.train.train import test_epoch   # <- test loop

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
    1: [], #[r"D:\Data\235", r"D:\Data\237"],
    2: [r"D:\Data\112"], # r"D:\Data\113", r"D:\Data\114", r"D:\Data\115", r"D:\Data\116"],
}
TEST_FAMILY_IDS = [2]
CKPT_DIR        = fr"D:\Data\model_checkpoints"    # where train_loop saved checkpoints
BATCH_SIZE      = 64
NUM_WORKERS     = 0                                # Windows-safe
LATENT_DIM      = 2
M_FIB           = 15                               # same latent grid density
SPECS_PER_FILE  = 100
TEST_SIZE       = 0.20
SPLIT_SEED      = 92

out_dir_fig = r"D:\Data\Figs"
os.makedirs(out_dir_fig, exist_ok=True)


# -----------------------------------------------
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


def build_loaders():
    # From one folder
    # (train_fns, test_fns), (train_ids, test_ids), specs_per_file = load_gerbils(
    #     DATA_ROOT,
    #     specs_per_file=SPECS_PER_FILE,
    #     families=TEST_FAMILY_IDS,
    #     test_size=TEST_SIZE,
    #     seed=SPLIT_SEED,
    #     check=True,
    # )
    # From multiple folders
    (train_fns, test_fns), (train_ids, test_ids), specs_per_file = load_gerbils_multi(
        gerbil_filepath=DATA_ROOT,
        specs_per_file=SPECS_PER_FILE,
        families=TEST_FAMILY_IDS,
        test_size=TEST_SIZE,
        seed=SPLIT_SEED,
        check=True,
    )

    # gets one sample at a time by index via __getitem__ and knows how many samples exist via __len__, each item is (spec,family_id) or (spec,c,family_id)
    test_ds = bird_data(test_fns, test_ids, specs_per_file=specs_per_file,
                        transform=spec_to_tensor, conditional=False) # conditional=False--> no arena info
    train_ds = bird_data(train_fns, train_ids, specs_per_file=specs_per_file,
                         transform=spec_to_tensor, conditional=False) # conditional=False--> no arena info


    # wraps the dataset to give mini-batches, splits by batch_size
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, test_loader, train_fns, test_fns


def rebuild_model(device):
    decoder = get_decoder_arch(dataset_name="gerbil_ava", latent_dim=LATENT_DIM)
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
    train_loader, test_loader,train_fns, test_fns = build_loaders()


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
            test_loader,
            latent_grid.to(device),
            qmc_loss_func,
            conditional=False,
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
    with torch.no_grad():
         model_grid_plot(model, n_samples_dim=20, origin="lower", cm="inferno",show=False,
                        fn=os.path.join(out_dir_fig, "decoder_grid.png"),)  # may take a bit


    # 6c) latent embeddings scatter
    # Test data embedding
    test_embeddings, test_labels = model.embed_data(
        latent_grid.to(device),
        test_loader,
        qmc_lp,
        embed_type="rqmc",
        n_samples=5,
    )

    # Train data embedding
    train_embeddings, train_labels = model.embed_data(
        latent_grid.to(device),
        train_loader,
        qmc_lp,
        embed_type="rqmc",
        n_samples=5,
    )


    # =========================
    # Plot — all families together
    # =========================

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

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
    import math

    def _grid_nrows_ncols(n, max_cols=4):
        cols = min(n, max_cols)
        rows = math.ceil(n / cols)
        return rows, cols

    families_present = sorted(np.unique(np.concatenate([train_labels, test_labels])))
    n = len(families_present)
    rows, cols = _grid_nrows_ncols(n, max_cols=4)

    # ---------- TRAIN panels ----------
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for idx, fam in enumerate(families_present):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        m_tr = (train_labels == fam)
        ax.scatter(train_embeddings[m_tr, 0], train_embeddings[m_tr, 1],
                   s=3, marker=".", c="C0", alpha=0.6, linewidths=0)
        ax = format_plot_axis(
            ax, xlim=(0, 1), ylim=(0, 1),
            xlabel="Latent dim 1", ylabel="Latent dim 2",
            title=f"Train — family {int(fam)}"
        )

    # hide any unused axes
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_fig, "embeddings_train_panels_by_family.png"),
                dpi=300, bbox_inches="tight")
    plt.show();
    plt.close(fig)

    # ---------- TEST panels ----------
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for idx, fam in enumerate(families_present):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        m_te = (test_labels == fam)
        ax.scatter(test_embeddings[m_te, 0], test_embeddings[m_te, 1],
                   s=3, marker=".", c="C0", alpha=0.6, linewidths=0)
        ax = format_plot_axis(
            ax, xlim=(0, 1), ylim=(0, 1),
            xlabel="Latent dim 1", ylabel="Latent dim 2",
            title=f"Test — family {int(fam)}"
        )

    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_fig, "embeddings_test_panels_by_family.png"),
                dpi=300, bbox_inches="tight")
    plt.show();
    plt.close(fig)

    # =========================
    # Plot - By location
    # =========================
    # Read locations from hdf5 files
    def _read_locations_for_file(h5_path):
        with h5py.File(h5_path, 'r') as f:
            locs = f['locations'][:]  # bytes array length = number of specs in this file
        # decode bytes -> str
        return np.array([x.decode('ASCII') for x in locs], dtype=object)

    def build_locations_vector(file_list):
        """
        Returns a 1D array of strings (arena_1/arena_2/underground) aligned
        with the order bird_data/test_loader iterate (file-major, no shuffle).
        """
        out = []
        for p in file_list:
            out.extend(_read_locations_for_file(p))
        return np.array(out, dtype=object)

    # locations
    train_locs = build_locations_vector(train_fns)  # shape = len(train_embeddings)
    test_locs = build_locations_vector(test_fns)  # shape = len(test_embeddings)

    #
    # plot - one family at one location (train vs test panels)
    def plot_family_location(fam, loc_name,
                             train_embeddings, train_labels, train_locs,
                             test_embeddings, test_labels, test_locs,
                             out_dir_fig):
        m_tr = (train_labels == fam) & (train_locs == loc_name)
        m_te = (test_labels == fam) & (test_locs == loc_name)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

        # Test
        ax = axes[0]
        ax.scatter(test_embeddings[m_te, 0], test_embeddings[m_te, 1],
                   s=3, marker=".", alpha=0.7, c="C0")
        ax = format_plot_axis(
            ax, xlim=(0, 1), ylim=(0, 1),
            xlabel="Latent dim 1", ylabel="Latent dim 2",
            title=f"Test — family {int(fam)}, {loc_name}"
        )

        # Train
        ax = axes[1]
        ax.scatter(train_embeddings[m_tr, 0], train_embeddings[m_tr, 1],
                   s=3, marker=".", alpha=0.7, c="C0")
        ax = format_plot_axis(
            ax, xlim=(0, 1), ylim=(0, 1),
            xlabel="Latent dim 1", ylabel="Latent dim 2",
            title=f"Train — family {int(fam)}, {loc_name}"
        )

        plt.tight_layout()
        fn = os.path.join(out_dir_fig, f"embeddings_family{int(fam)}_{loc_name}_train_vs_test.png")
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        plt.close(fig)

    for fam in sorted(np.unique(np.concatenate([train_labels, test_labels]))):
        for loc in ["arena_1", "arena_2", "underground"]:
            plot_family_location(fam, loc,
                                     train_embeddings, train_labels, train_locs,
                                     test_embeddings, test_labels, test_locs,
                                     out_dir_fig)

    # plot - “all families pooled” but filtered to a single location
    def plot_all_families_at_location(loc_name,
                                      train_embeddings, train_locs,
                                      test_embeddings, test_locs,
                                      out_dir_fig):
        m_tr = (train_locs == loc_name)
        m_te = (test_locs == loc_name)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

        # Test
        ax = axes[0]
        ax.scatter(test_embeddings[m_te, 0], test_embeddings[m_te, 1],
                   s=3, marker=".", alpha=0.7, c="C0")
        ax = format_plot_axis(
            ax, xlim=(0, 1), ylim=(0, 1),
            xlabel="Latent dim 1", ylabel="Latent dim 2",
            title=f"Test — all families @ {loc_name}"
        )

        # Train
        ax = axes[1]
        ax.scatter(train_embeddings[m_tr, 0], train_embeddings[m_tr, 1],
                   s=3, marker=".", alpha=0.7, c="C0")
        ax = format_plot_axis(
            ax, xlim=(0, 1), ylim=(0, 1),
            xlabel="Latent dim 1", ylabel="Latent dim 2",
            title=f"Train — all families @ {loc_name}"
        )

        plt.tight_layout()
        fn = os.path.join(out_dir_fig, f"embeddings_allfamilies_{loc_name}_train_vs_test.png")
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        plt.close(fig)

    for loc in ["arena_1", "arena_2", "underground"]:
        plot_all_families_at_location(loc, train_embeddings, train_locs,
                                      test_embeddings, test_locs,
                                      out_dir_fig)

    # =========================
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
