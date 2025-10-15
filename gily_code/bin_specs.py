
# ------------------ simple bin-count dump plugin ------------------
# fast_bin_dump_counts.py
# -------------------------------------------------------------
# Stand-alone pre-training checker.
# For each spectrogram clip:
#   • threshold magnitudes
#   • count active pixels in LOW / MID / HIGH bands across all frames
#   • apply MID-priority rule, else pick argmax of band counts
# Saves every clip PNG into its decided bin folder and a SUM.png per bin.
# -------------------------------------------------------------

import os, glob
import h5py
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
# Where to find data: <root>/processed-data/family{ID}/*.hdf5
ROOTS_PER_FAMILY = {
    # 1: [r"D:\Data\235", r"D:\Data\237"],
    1: [r"D:\Data\237"],
}
FAMILIES = [1]                        # which families to include
OUT_DIR  = r"D:\Data\Figs_fastdump_counts"
os.makedirs(OUT_DIR, exist_ok=True)

# Spectrogram frequency axis (matches your preprocessing)
MIN_FREQ_HZ    = 500
MAX_FREQ_HZ    = 62500
NUM_FREQ_BINS  = 128
FREQ_AXIS_HZ   = np.linspace(MIN_FREQ_HZ, MAX_FREQ_HZ, NUM_FREQ_BINS, dtype=np.float64)

# Band edges (LOW, MID, HIGH).
EDGES_HZ = (18_000.0, 26_000.0)

# Thresholding / gating / decision
NOISE_FLOOR_PERCENTILE = 70.0   # percentile of all magnitudes in the clip to define "active" threshold
MIN_FRAME_ACTIVE_RATIO = 0.01   # keep frames whose energy >= this * median(frame energy)
MIN_ACTIVE_PIXELS      = 50     # require at least this many active pixels overall to consider the clip valid
MID_PRIORITY_FRAC      = 0.25   # if MID / total >= this, force MID (clip-level priority)
TIE_MARGIN             = 0.02   # if top two band fracs within ±2%, choose MID > HIGH > LOW (stable tie-break)

# Visualization
CMAP          = "magma"
DPI_INDIV     = 180
DPI_SUM       = 200
FIGSIZE_INDIV = (6.0, 4.6)
FIGSIZE_SUM   = (6.4, 4.8)

# =========================
# Helpers
# =========================



def safe_name(text: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(text))

def list_family_files(roots_per_family: dict, families: list[int]) -> list[str]:
    """Collect *.hdf5 under {root}/processed-data/family{fam} for all families."""
    files = []
    for fam in families:
        roots = roots_per_family.get(fam, [])
        for root in roots:
            spec_dir = os.path.join(root, "processed-data", f"family{fam}")
            files.extend(glob.glob(os.path.join(spec_dir, "*.hdf5")))
    return sorted(list(set(files)))

def band_indices(freq_axis_hz: np.ndarray, edges_hz: tuple[float, float]):
    """
    Return index arrays (low_idx, mid_idx, high_idx) for LOW/MID/HIGH.
    - LOW: f < e1
    - MID: e1 <= f < e2
    - HIGH: f >= e2
    """
    e1, e2 = edges_hz
    low_idx  = np.where(freq_axis_hz < e1)[0]
    mid_idx  = np.where((freq_axis_hz >= e1) & (freq_axis_hz < e2))[0]
    high_idx = np.where(freq_axis_hz >= e2)[0]
    return low_idx, mid_idx, high_idx

def resize_to(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize spectrogram (F,T) to target (F,T) with bilinear-ish zoom."""
    F0, T0 = arr.shape
    Ft, Tt = target_shape
    if (F0, T0) == (Ft, Tt):
        return arr
    return zoom(arr, (Ft / max(F0, 1), Tt / max(T0, 1)), order=1)

def frame_gate_mask(spec_FT: np.ndarray, min_ratio: float) -> np.ndarray:
    """
    Per-frame energy gate:
    keep frame t if sum(spec[:,t]) >= min_ratio * median_t sum(spec[:,t]).
    Returns boolean mask of shape (T,).
    """
    e_t = np.maximum(spec_FT, 0.0).sum(axis=0)
    med = np.median(e_t) + 1e-12
    return e_t >= (min_ratio * med)

def save_spec_png(spec_FT: np.ndarray,
                  out_png: str,
                  *,
                  freq_axis_hz: np.ndarray,
                  edges_hz: tuple[float, float],
                  title: str | None = None,
                  subtitle: str | None = None,
                  cmap: str = CMAP,
                  dpi: int = DPI_INDIV,
                  figsize: tuple[float, float] = FIGSIZE_INDIV):
    """Plot spectrogram with dashed band edges."""
    F, T = spec_FT.shape
    y0_khz = float(freq_axis_hz.min()) / 1000.0
    y1_khz = float(freq_axis_hz.max()) / 1000.0
    extent = [0, T, y0_khz, y1_khz]

    vmin = np.percentile(spec_FT, 5)
    vmax = np.percentile(spec_FT, 95)
    if vmin == vmax:
        vmin, vmax = float(spec_FT.min()), float(spec_FT.max())

    plt.figure(figsize=figsize, dpi=dpi)
    im = plt.imshow(
        spec_FT, origin="lower", aspect="auto", extent=extent,
        cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
    )
    plt.colorbar(im, pad=0.01, shrink=0.9)

    # Draw band edges
    e1k, e2k = edges_hz[0] / 1000.0, edges_hz[1] / 1000.0
    plt.axhline(e1k, color="white", linestyle="--", linewidth=1.0, alpha=0.9)
    plt.axhline(e2k, color="white", linestyle="--", linewidth=1.0, alpha=0.9)

    if title:
        plt.title(title, fontsize=10)
    if subtitle:
        ax = plt.gca()
        ax.text(0.02, 0.98, subtitle, transform=ax.transAxes,
                fontsize=9, color="w", va="top", ha="left",
                bbox=dict(facecolor="0.05", alpha=0.55, pad=3, edgecolor="none"))

    plt.xlabel("Time (frames)")
    plt.ylabel("Frequency (kHz)")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def mask_and_select_bands(spec_FT: np.ndarray, mask_arr: np.ndarray,
                          row_band_idxs: np.ndarray) -> np.ndarray:
    cols = np.any(mask_arr[row_band_idxs], axis=0)  # keep only columns that have any True in low rows
    selected_arr = spec_FT[row_band_idxs][:, cols]
    return selected_arr

# =========================
# Core: count-all binning
# =========================
def count_all_bins_decision(spec_FT: np.ndarray,
                            freq_axis_hz: np.ndarray,
                            edges_hz: tuple[float, float],
                            *,
                            noise_floor_percentile: float = NOISE_FLOOR_PERCENTILE,
                            min_frame_active_ratio: float = MIN_FRAME_ACTIVE_RATIO,
                            min_active_pixels: int = MIN_ACTIVE_PIXELS,
                            mid_priority_frac: float = MID_PRIORITY_FRAC,
                            tie_margin: float = TIE_MARGIN):
    """
    Steps:
      1) Gate frames by per-frame energy (robust to silence/noise).
      2) Compute a clip-specific magnitude threshold as given percentile.
      3) Count active pixels (>= threshold) in LOW/MID/HIGH across KEPT frames.
      4) Decide:
           - If total active < min_active_pixels -> default LOW
           - Else if MID / total >= mid_priority_frac -> MID
           - Else choose argmax of band counts, tie-broken by tie_margin favoring MID>HIGH>LOW.
    Returns: counts (3,), fracs (3,), winner (0/1/2), rule (str).
    """
    S = np.asarray(spec_FT, dtype=np.float64)
    S = np.maximum(S, 0.0)
    F, T = S.shape

    # 1) gate frames
    keep = frame_gate_mask(S, min_ratio=min_frame_active_ratio)

    # 2) threshold
    thr = np.percentile(S, noise_floor_percentile)  # scalar threshold for "active"
    if not np.isfinite(thr):
        thr = 0.0

    # 3) count active pixels per band across kept frames
    low_idx, mid_idx, high_idx = band_indices(freq_axis_hz, edges_hz)

    # Boolean mask of active pixels
    A = (S >= thr)
    if keep.any():
        A[:, ~keep] = False  # zero-out dropped frames

    low_part = mask_and_select_bands(S, A, low_idx)
    mid_part = mask_and_select_bands(S, A, mid_idx)
    high_part = mask_and_select_bands(S, A, high_idx)

    frame_count_low = np.count_nonzero(np.count_nonzero(low_part > 0, axis=0))
    frame_count_mid = np.count_nonzero(np.count_nonzero(mid_part > 0, axis=0))
    frame_count_high = np.count_nonzero(np.count_nonzero(high_part > 0, axis=0))

    frame_count_low2 = np.sum(low_part, axis=0)
    frame_count_mid2 = np.sum(mid_part, axis=0)
    frame_count_high2 = np.sum(high_part, axis=0)

    cL = int(A[low_idx, :].sum())   if low_idx.size  else 0
    cM = int(A[mid_idx, :].sum())   if mid_idx.size  else 0
    cH = int(A[high_idx, :].sum())  if high_idx.size else 0

    counts = np.array([cL, cM, cH], dtype=int)
    total  = int(counts.sum())
    fracs  = counts / (total if total > 0 else 1)

    # 4) decision
    if total < min_active_pixels:
        return counts, fracs, 0, "too_few_active_pixels"

    if fracs[1] >= mid_priority_frac:
        return counts, fracs, 1, "mid_priority_clip"

    # argmax with tie-breaks within margin
    order = np.argsort(-counts)  # descending
    top, second = order[0], order[1]
    if total > 0:
        if (counts[top] - counts[second]) / max(counts[top], 1) <= tie_margin:
            # prefer MID > HIGH > LOW on near-ties
            for pref in (1, 2, 0):
                if counts[pref] == counts[top] or \
                   abs(counts[pref] - counts[top]) / max(counts[top],1) <= tie_margin:
                    return counts, fracs, int(pref), "tie_break_pref"
    return counts, fracs, int(top), "argmax_clip"

# =========================
# Main dump
# =========================
def main():
    files = list_family_files(ROOTS_PER_FAMILY, FAMILIES)
    if not files:
        print("[WARN] No .hdf5 files found. Check ROOTS_PER_FAMILY and FAMILIES.")
        return

    # Create bin dirs
    bin_dirs = {
        0: os.path.join(OUT_DIR, "bin_0_low"),
        1: os.path.join(OUT_DIR, "bin_1_mid"),
        2: os.path.join(OUT_DIR, "bin_2_high"),
    }
    for d in bin_dirs.values():
        os.makedirs(d, exist_ok=True)

    # SUM accumulators (normalized per clip before adding)
    sum_accum = {0: None, 1: None, 2: None}  # np.array(F,T)
    sum_shape = {0: None, 1: None, 2: None}
    running_idx = {0: 0, 1: 0, 2: 0}

    # Count total specs (optional progress info)
    total_specs = 0
    for fn in files:
        with h5py.File(fn, "r") as f:
            total_specs += len(f.get("specs", []))
    print(f"[INFO] Processing {len(files)} file(s); ~{total_specs} spectrograms...")

    for fn in tqdm(files):
        with h5py.File(fn, "r", locking=False) as f:
            specs = f["specs"]            # (N, F, T)
            N = len(specs)
            for i in range(N):
                spec = np.array(specs[i], dtype=np.float64)  # (F,T)

                counts, fracs, winner, rule = count_all_bins_decision(
                    spec,
                    freq_axis_hz=FREQ_AXIS_HZ,
                    edges_hz=EDGES_HZ,
                    noise_floor_percentile=NOISE_FLOOR_PERCENTILE,
                    min_frame_active_ratio=MIN_FRAME_ACTIVE_RATIO,
                    min_active_pixels=MIN_ACTIVE_PIXELS,
                    mid_priority_frac=MID_PRIORITY_FRAC,
                    tie_margin=TIE_MARGIN,
                )

                b = int(winner)  # 0/1/2
                cL, cM, cH = (int(counts[0]), int(counts[1]), int(counts[2]))
                fL, fM, fH = (float(fracs[0]), float(fracs[1]), float(fracs[2]))
                subtitle = f"L/M/H={cL}/{cM}/{cH}  frac=[{fL:.2f},{fM:.2f},{fH:.2f}]  → bin {b} ({rule})"

                # Save individual PNG
                out_png = os.path.join(
                    bin_dirs[b],
                    safe_name(f"{os.path.splitext(os.path.basename(fn))[0]}_idx{i:05d}.png")
                )
                save_spec_png(
                    spec,
                    out_png,
                    freq_axis_hz=FREQ_AXIS_HZ,
                    edges_hz=EDGES_HZ,
                    title=None,
                    subtitle=subtitle,
                    cmap=CMAP,
                    dpi=DPI_INDIV,
                    figsize=FIGSIZE_INDIV,
                )

                # Add to SUM (normalize per item)
                tgt = sum_shape[b]
                if tgt is None:
                    tgt = spec.shape
                    sum_shape[b] = tgt
                    sum_accum[b] = np.zeros(tgt, dtype=np.float32)

                r = resize_to(spec, tgt).astype(np.float32)
                mx = float(r.max())
                if mx > 0:
                    r = r / mx
                sum_accum[b] += r
                running_idx[b] += 1

    # Save SUM images & raw npy
    for b in (0, 1, 2):
        if sum_accum[b] is None:
            print(f"[INFO] Bin {b} had no items; skipping SUM.")
            continue
        sum_png = os.path.join(bin_dirs[b], "SUM_spectrogram.png")
        save_spec_png(
            sum_accum[b],
            sum_png,
            freq_axis_hz=FREQ_AXIS_HZ,
            edges_hz=EDGES_HZ,
            title=f"SUM — bin {b}",
            subtitle=None,
            cmap=CMAP,
            dpi=DPI_SUM,
            figsize=FIGSIZE_SUM,
        )
        np.save(os.path.join(bin_dirs[b], "SUM_spectrogram.npy"), sum_accum[b])
        print(f"[INFO] Bin {b}: saved SUM to {sum_png}  (count={running_idx[b]})")

    print("[DONE] All PNGs and SUMs written to:", OUT_DIR)

if __name__ == "__main__":
    main()



######################################################################
# ARG MAX VERSION
######################################################################

# # fast_bin_dump.py
# # -------------------------------------------------------------
# # Stand-alone analysis script (no training required).
# # Groups spectrograms into LOW/MID/HIGH via per-frame band energy
# # with MID-priority, saves all PNGs, and also SUM image per bin.
# # -------------------------------------------------------------
#
# import os, glob
# import h5py
# import numpy as np
# from tqdm import tqdm
# from scipy.ndimage import zoom
# import matplotlib.pyplot as plt
#
# # =========================
# # CONFIG
# # =========================
# # Where to find data (like your previous structure: <root>/processed-data/family{ID}/*.hdf5)
# ROOTS_PER_FAMILY = {
#     # 1: [r"D:\Data\235", r"D:\Data\237"],
#     1: [r"D:\Data\237"],
#     # 2: [r"D:\Data\113", r"D:\Data\114", r"D:\Data\115", r"D:\Data\116"],
# }
# FAMILIES = [1]                # which families to include
# OUT_DIR  = r"D:\Data\Figs_fastdump"  # where PNGs and SUMs go
# os.makedirs(OUT_DIR, exist_ok=True)
#
# # Spectrogram frequency axis (matches your preprocessing)
# MIN_FREQ_HZ    = 500
# MAX_FREQ_HZ    = 62500
# NUM_FREQ_BINS  = 128
# FREQ_AXIS_HZ   = np.linspace(MIN_FREQ_HZ, MAX_FREQ_HZ, NUM_FREQ_BINS, dtype=np.float64)
#
# # Band edges (LOW, MID, HIGH). Example: LOW < 22k, MID [22k,27k), HIGH >= 27k
# EDGES_HZ = (18_000.0, 27_000.0)
#
# # Gating / decision parameters
# ALPHA_GATE            = 1e-4   # frame energy >= ALPHA * median(frame energy) to be considered
# MID_PRIORITY_FRAC     = 0.25   # if MID fraction >= this, label clip as MID regardless of HIGH
# USE_AMBIGUOUS         = False  # if True, add a 4th "ambiguous" bucket when band fractions are all similar
# AMBIGUOUS_MARGIN      = 0.10   # max frac spread to consider "all comparable" (LOW/MID/HIGH similar)
#
# # Visualization
# CMAP          = "magma"
# DPI_INDIV     = 180
# DPI_SUM       = 200
# FIGSIZE_INDIV = (6.0, 4.6)
# FIGSIZE_SUM   = (6.4, 4.8)
#
# # =========================
# # Helpers
# # =========================
#
# def safe_name(text: str) -> str:
#     return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(text))
#
# def list_family_files(roots_per_family: dict, families: list[int]) -> list[str]:
#     """Collect *.hdf5 under {root}/processed-data/family{fam} for all families."""
#     files = []
#     for fam in families:
#         roots = roots_per_family.get(fam, [])
#         for root in roots:
#             spec_dir = os.path.join(root, "processed-data", f"family{fam}")
#             files.extend(glob.glob(os.path.join(spec_dir, "*.hdf5")))
#     return sorted(list(set(files)))
#
# def frame_gate_mask(spec_FT: np.ndarray, alpha: float) -> np.ndarray:
#     """
#     Per-frame energy gate:
#     keep frame t if sum(spec[:,t]) >= alpha * median_t sum(spec[:,t]).
#     Returns boolean mask of shape (T,).
#     """
#     e_t = np.maximum(spec_FT, 0.0).sum(axis=0)
#     med = np.median(e_t) + 1e-12
#     return e_t >= (alpha * med)
#
# def band_indices(freq_axis_hz: np.ndarray, edges_hz: tuple[float, float]):
#     """
#     Return index slices (low_idx, mid_idx, high_idx) for LOW/MID/HIGH.
#     - LOW: f < e1
#     - MID: e1 <= f < e2
#     - HIGH: f >= e2
#     """
#     e1, e2 = edges_hz
#     low_idx  = np.where(freq_axis_hz < e1)[0]
#     mid_idx  = np.where((freq_axis_hz >= e1) & (freq_axis_hz < e2))[0]
#     high_idx = np.where(freq_axis_hz >= e2)[0]
#     return low_idx, mid_idx, high_idx
#
# def per_frame_band_votes(spec_FT: np.ndarray,
#                          freq_axis_hz: np.ndarray,
#                          edges_hz: tuple[float, float],
#                          alpha_gate: float,
#                          mid_priority_frac: float,
#                          use_ambiguous: bool,
#                          ambiguous_margin: float):
#     """
#     Core logic:
#       1) Gate frames by per-frame energy.
#       2) For each KEPT frame, sum energy in LOW/MID/HIGH bands.
#       3) Per-frame vote:
#            - If MID / total >= mid_priority_frac  -> vote MID
#            - Else vote argmax([E_low, E_mid, E_high])
#       4) Aggregate votes across frames → final counts + fractions + decision.
#
#     Returns:
#       counts (np.ndarray int[3] or [4]),
#       fracs  (np.ndarray float[3] or [4]),
#       winner (int),
#       frame_track_hz (np.ndarray[T] with band center frequency for the winning band per kept frame; NaN for dropped)
#       rule   (str) describing which rule finalized the decision
#     """
#     S = np.asarray(spec_FT, dtype=np.float64)
#     F, T = S.shape
#     gate = frame_gate_mask(S, alpha_gate)  # (T,)
#
#     low_idx, mid_idx, high_idx = band_indices(freq_axis_hz, edges_hz)
#     # Precompute band center frequencies (for plotting a simple track)
#     def _band_center_hz(idx):
#         return float(freq_axis_hz[idx].mean()) if idx.size else np.nan
#     low_c  = _band_center_hz(low_idx)
#     mid_c  = _band_center_hz(mid_idx)
#     high_c = _band_center_hz(high_idx)
#
#     band_center_by_id = {0: low_c, 1: mid_c, 2: high_c}
#
#     # Track the per-frame winning band (for overlay)
#     frame_track_hz = np.full((T,), np.nan, dtype=np.float64)
#
#     votes = []
#     for t in range(T):
#         if not gate[t]:
#             continue
#         col = S[:, t]
#         E_low  = float(col[low_idx].sum())  if low_idx.size  else 0.0
#         E_mid  = float(col[mid_idx].sum())  if mid_idx.size  else 0.0
#         E_high = float(col[high_idx].sum()) if high_idx.size else 0.0
#         total = E_low + E_mid + E_high
#
#         if total <= 0:
#             # no energy, ignore the frame
#             continue
#
#         frac_mid = E_mid / total
#         if frac_mid >= mid_priority_frac:
#             b = 1  # MID
#             rule = "mid_priority_frame"
#         else:
#             # argmax LOW/MID/HIGH
#             b = int(np.argmax([E_low, E_mid, E_high]))
#             rule = "argmax_frame"
#
#         votes.append(b)
#         frame_track_hz[t] = band_center_by_id[b]
#
#     # Aggregate
#     if len(votes) == 0:
#         # No valid frames -> default LOW or ambiguous
#         if use_ambiguous:
#             counts = np.array([0, 0, 0, 1], dtype=int)  # force ambiguous
#             fracs  = counts / counts.sum()
#             return counts, fracs, 3, frame_track_hz, "no_valid_frames"
#         counts = np.array([1, 0, 0], dtype=int)
#         fracs  = counts / counts.sum()
#         return counts, fracs, 0, frame_track_hz, "no_valid_frames"
#
#     votes = np.array(votes, dtype=int)
#     counts = np.bincount(votes, minlength=3)
#     total_votes = int(counts.sum())
#     fracs = counts / (total_votes if total_votes > 0 else 1)
#
#     # Optional "ambiguous" (LOW/MID/HIGH all similar)
#     if use_ambiguous:
#         spread = float(np.max(fracs) - np.min(fracs))
#         if spread <= ambiguous_margin:
#             counts4 = np.concatenate([counts, np.array([1], dtype=int)])  # put 1 vote in ambiguous
#             fracs4  = counts4 / counts4.sum()
#             return counts4, fracs4, 3, frame_track_hz, "ambiguous_all_comparable"
#
#     # Final decision
#     # If clip-level MID fraction exceeds threshold, mark MID; else argmax
#     if fracs[1] >= mid_priority_frac:
#         return counts, fracs, 1, frame_track_hz, "mid_priority_clip"
#     winner = int(np.argmax(counts))
#     return counts, fracs, winner, frame_track_hz, "argmax_clip"
#
# def resize_to(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
#     """Resize spectrogram (F,T) to target (F,T) with bilinear-ish zoom."""
#     F0, T0 = arr.shape
#     Ft, Tt = target_shape
#     if (F0, T0) == (Ft, Tt):
#         return arr
#     return zoom(arr, (Ft / max(F0, 1), Tt / max(T0, 1)), order=1)
#
# def save_spec_png(spec_FT: np.ndarray,
#                   out_png: str,
#                   *,
#                   freq_axis_hz: np.ndarray,
#                   edges_hz: tuple[float, float],
#                   frame_track_hz: np.ndarray | None = None,
#                   title: str | None = None,
#                   subtitle: str | None = None,
#                   cmap: str = CMAP,
#                   dpi: int = DPI_INDIV,
#                   figsize: tuple[float, float] = FIGSIZE_INDIV):
#     """Plot spectrogram with dashed band edges and (optional) per-frame winning-band track."""
#     F, T = spec_FT.shape
#     y0_khz = float(freq_axis_hz.min()) / 1000.0
#     y1_khz = float(freq_axis_hz.max()) / 1000.0
#     extent = [0, T, y0_khz, y1_khz]
#
#     vmin = np.percentile(spec_FT, 5)
#     vmax = np.percentile(spec_FT, 95)
#     if vmin == vmax:
#         vmin, vmax = float(spec_FT.min()), float(spec_FT.max())
#
#     plt.figure(figsize=figsize, dpi=dpi)
#     im = plt.imshow(
#         spec_FT, origin="lower", aspect="auto", extent=extent,
#         cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
#     )
#     plt.colorbar(im, pad=0.01, shrink=0.9)
#
#     # Draw band edges
#     e1k, e2k = edges_hz[0] / 1000.0, edges_hz[1] / 1000.0
#     plt.axhline(e1k, color="white", linestyle="--", linewidth=1.0, alpha=0.9)
#     plt.axhline(e2k, color="white", linestyle="--", linewidth=1.0, alpha=0.9)
#
#     # Overlay per-frame band track
#     if frame_track_hz is not None:
#         t = np.arange(len(frame_track_hz), dtype=np.float64)
#         fkhz = frame_track_hz / 1000.0
#         m = np.isfinite(fkhz)
#         if np.any(m):
#             plt.plot(t[m], fkhz[m], linewidth=1.2, alpha=0.95)
#
#     if title:
#         plt.title(title, fontsize=10)
#     if subtitle:
#         ax = plt.gca()
#         ax.text(0.02, 0.98, subtitle, transform=ax.transAxes,
#                 fontsize=9, color="w", va="top", ha="left",
#                 bbox=dict(facecolor="0.05", alpha=0.55, pad=3, edgecolor="none"))
#
#     plt.xlabel("Time (frames)")
#     plt.ylabel("Frequency (kHz)")
#     plt.tight_layout()
#     plt.savefig(out_png, bbox_inches="tight")
#     plt.close()
#
# # =========================
# # Main dump
# # =========================
#
# def main():
#     files = list_family_files(ROOTS_PER_FAMILY, FAMILIES)
#     if not files:
#         print("[WARN] No .hdf5 files found. Check ROOTS_PER_FAMILY and FAMILIES.")
#         return
#
#     # Create bin dirs
#     bin_dirs = {
#         0: os.path.join(OUT_DIR, "bin_0_low"),
#         1: os.path.join(OUT_DIR, "bin_1_mid"),
#         2: os.path.join(OUT_DIR, "bin_2_high"),
#     }
#     for d in bin_dirs.values():
#         os.makedirs(d, exist_ok=True)
#
#     # First pass: collect shapes per bin (to choose SUM target shapes)
#     shapes_per_bin = {0: [], 1: [], 2: []}
#
#     print(f"[INFO] Scanning files to preview shapes ({len(files)} files)...")
#     total_specs = 0
#     for fn in tqdm(files):
#         with h5py.File(fn, "r") as f:
#             S = f["specs"]
#             total_specs += len(S)
#             if len(S) == 0:  # just in case
#                 continue
#     if total_specs == 0:
#         print("[WARN] No spectrograms found in any file.")
#         return
#
#     # We’ll build the SUM incrementally; we need to know target (F,T).
#     # We’ll do it on-the-fly per bin the first time we see a clip for that bin:
#     sum_accum = {0: None, 1: None, 2: None}  # np.array(F,T)
#     sum_shape = {0: None, 1: None, 2: None}
#
#     running_idx = {0: 0, 1: 0, 2: 0}
#
#     print(f"[INFO] Processing {len(files)} files; total spectrograms ≈ {total_specs}...")
#     for fn in tqdm(files):
#         with h5py.File(fn, "r", locking=False) as f:
#             specs = f["specs"]            # (N, F, T)
#             N = len(specs)
#             for i in range(N):
#                 spec = np.array(specs[i], dtype=np.float64)  # (F,T)
#
#                 # --- Decide bin via per-frame band votes ---
#                 counts, fracs, winner, track_hz, rule = per_frame_band_votes(
#                     spec,
#                     freq_axis_hz=FREQ_AXIS_HZ,
#                     edges_hz=EDGES_HZ,
#                     alpha_gate=ALPHA_GATE,
#                     mid_priority_frac=MID_PRIORITY_FRAC,
#                     use_ambiguous=USE_AMBIGUOUS,
#                     ambiguous_margin=AMBIGUOUS_MARGIN,
#                 )
#
#                 b = int(winner)  # 0/1/2   (we ignore ambiguous=3 here to keep 3 bins)
#                 cL, cM, cH = (int(counts[0]), int(counts[1]), int(counts[2]))
#                 fL, fM, fH = (float(fracs[0]), float(fracs[1]), float(fracs[2]))
#                 subtitle = f"L/M/H={cL}/{cM}/{cH}  frac=[{fL:.2f},{fM:.2f},{fH:.2f}]  → bin {b} ({rule})"
#
#                 # --- Save individual PNG ---
#                 out_png = os.path.join(
#                     bin_dirs[b],
#                     safe_name(f"{os.path.splitext(os.path.basename(fn))[0]}_idx{i:05d}.png")
#                 )
#                 save_spec_png(
#                     spec,
#                     out_png,
#                     freq_axis_hz=FREQ_AXIS_HZ,
#                     edges_hz=EDGES_HZ,
#                     frame_track_hz=track_hz,
#                     title=None,
#                     subtitle=subtitle,
#                     cmap=CMAP,
#                     dpi=DPI_INDIV,
#                     figsize=FIGSIZE_INDIV,
#                 )
#
#                 # --- Add to SUM (normalize per item to avoid dominance) ---
#                 tgt = sum_shape[b]
#                 if tgt is None:
#                     # Set target size to the FIRST encountered in this bin; or use max strategy if preferred
#                     tgt = spec.shape
#                     sum_shape[b] = tgt
#                     sum_accum[b] = np.zeros(tgt, dtype=np.float32)
#
#                 r = resize_to(spec, tgt).astype(np.float32)
#                 mx = float(r.max())
#                 if mx > 0:
#                     r = r / mx
#                 sum_accum[b] += r
#                 running_idx[b] += 1
#
#     # --- Save SUM images & raw npy ---
#     for b in (0, 1, 2):
#         if sum_accum[b] is None:
#             print(f"[INFO] Bin {b} had no items; skipping SUM.")
#             continue
#         sum_png = os.path.join(bin_dirs[b], "SUM_spectrogram.png")
#         save_spec_png(
#             sum_accum[b],
#             sum_png,
#             freq_axis_hz=FREQ_AXIS_HZ,
#             edges_hz=EDGES_HZ,
#             frame_track_hz=None,
#             title=f"SUM — bin {b}",
#             subtitle=None,
#             cmap=CMAP,
#             dpi=DPI_SUM,
#             figsize=FIGSIZE_SUM,
#         )
#         np.save(os.path.join(bin_dirs[b], "SUM_spectrogram.npy"), sum_accum[b])
#         print(f"[INFO] Bin {b}: saved SUM to {sum_png}  (count={running_idx[b]})")
#
#     print("[DONE] All PNGs and SUMs written to:", OUT_DIR)
#
#
# if __name__ == "__main__":
#     main()
