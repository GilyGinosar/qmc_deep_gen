import numpy as np
from scipy.ndimage import binary_dilation, median_filter
import os
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

# Spectrogram frequency axis (matches your preprocessing)
MIN_FREQ_HZ    = 500
MAX_FREQ_HZ    = 62500
NUM_FREQ_BINS  = 128
FREQ_AXIS_HZ   = np.linspace(MIN_FREQ_HZ, MAX_FREQ_HZ, NUM_FREQ_BINS, dtype=np.float64)
FREQ_AXIS_KHZ = FREQ_AXIS_HZ / 1000
# LOW_BAND=(4,16)
# ALARM_LOW=(18,26)
# ALARM_HIGH=(40,55)
# HIGH_BAND=(28,60)

# Dataset names inside the HDF5:
DS_SPECS = "specs"        # (N, F, T) or (F, T)
DS_FREQS = "freqs_khz"    # (F,) in kHz; if you only have Hz, set to "freqs_hz" below
DS_FREQS_HZ = None        # e.g., "freqs_hz" if your file stores Hz (we'll convert)

#---

# ======== CONFIG ========
INPUT = Path(r"D:\Data\235\processed-data\family1")   # HDF5 with datasets described below
OUT   = Path(r"D:\Data\bin_previews_235")           # will create low/, alarm/, high/ inside
MAX_ITEMS = None                                 # limit for quick tests (e.g., 500) or None
FIG_DPI = 140



# Optional: draw dashed lines at these band edges (kHz) just for visualization
VIZ_BANDS = dict(
    mid_low=18.0,
    mid_high=26.0,
    hi_low=40.0,
    hi_high=55.0,
)

#---

# Visualization
CMAP          = "magma"
DPI_INDIV     = 180
DPI_SUM       = 200
FIGSIZE_INDIV = (6.0, 4.6)
FIGSIZE_SUM   = (6.4, 4.8)


def band_idx(freqs_khz, lo, hi):
    f = np.asarray(freqs_khz)
    return (f >= lo) & (f < hi)

def adaptive_activity_mask(S, eps=1e-12, db=True, col_q=0.9, add_db=6.0):
    """
    Per-column threshold:
    - Take percentile (e.g., 90th) within the column as a baseline
    - Mark pixels above (baseline + add_db) as active
    """
    X = np.asarray(S, float)
    if db:
        X = 20*np.log10(np.maximum(X, eps))
    # column-wise baseline
    base = np.percentile(X, col_q*100, axis=0, keepdims=True)  # shape (F, T)
    M = X > (base + add_db)
    # small temporal smoothing reduces salt/pepper
    M = median_filter(M.astype(np.uint8), size=(1, 3)).astype(bool)
    return M
def classify_spectrogram_three_way(
    S, freqs_khz, *,
    low_band=(4, 14),          # gerbil low noise band
    alarm_low=(18, 26),        # alarm fundamental
    alarm_high=(40, 55),       # alarm 2nd harmonic
    high_band=(28, 60),        # high-only calls
    co_win=1,                  # allow ±1 frame for co-occ
    min_frames_alarm=6,        # absolute frames
    min_frac_alarm=0.06,       # fraction of T
    min_frac_high=0.12,        # high-only fraction
    min_frac_low=0.12,         # low fraction
    # ---- new tiny knobs (safe defaults) ----
    near_tie_eps=0.03,         # if |frac_high-frac_low| ≤ eps → favor HIGH
    min_high_streak=3          # lock HIGH if ≥ this many consecutive high frames
):
    """
    Returns:
      label in {"low","alarm","high"} and a diagnostics dict.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation

    F, T = S.shape
    A = adaptive_activity_mask(S)  # boolean (F, T)

    # band masks (row indices)
    low_idx   = band_idx(freqs_khz, *low_band)
    aL_idx    = band_idx(freqs_khz, *alarm_low)
    aH_idx    = band_idx(freqs_khz, *alarm_high)
    high_idx  = band_idx(freqs_khz, *high_band)

    # per-column hits
    low_hits  = np.count_nonzero(A[low_idx],  axis=0)    # (T,)
    aL_hits   = np.count_nonzero(A[aL_idx],   axis=0)
    aH_hits   = np.count_nonzero(A[aH_idx],   axis=0)
    high_hits = np.count_nonzero(A[high_idx], axis=0)

    # per-frame activity booleans
    low_on   = (low_hits  > 0)
    aL_on    = (aL_hits   > 0)
    aH_on    = (aH_hits   > 0)
    high_on  = (high_hits > 0)

    # co-occurrence: dilate in time so near-simultaneous counts match
    if co_win > 0:
        st = (1, 2*co_win + 1)  # (freq, time)
        aL_act = binary_dilation(A[aL_idx], structure=np.ones(st)).any(axis=0)
        aH_act = binary_dilation(A[aH_idx], structure=np.ones(st)).any(axis=0)
    else:
        aL_act, aH_act = aL_on, aH_on

    alarm_co = aL_act & aH_act

    # fractions across time
    def _frac(x):  # safe for T==0
        return float(np.mean(x)) if T else 0.0

    frac_low   = _frac(low_on)
    frac_aL    = _frac(aL_on)
    frac_aH    = _frac(aH_on)
    frac_high  = _frac(high_on)
    frac_co    = _frac(alarm_co)

    n_co = int(alarm_co.sum())

    # helper: longest consecutive True run
    def _longest_streak(x_bool: np.ndarray) -> int:
        if T == 0 or not np.any(x_bool):
            return 0
        # sentinel False at both ends → diffs find starts/ends
        z = np.concatenate(([False], x_bool, [False]))
        runs = np.flatnonzero(z[1:] & ~z[:-1])
        rune = np.flatnonzero(~z[1:] & z[:-1])
        return int((rune - runs).max()) if runs.size else 0

    high_streak = _longest_streak(high_on)
    low_streak  = _longest_streak(low_on)

    # --- Decision rules (ordered) ---
    rule = None

    # 1) Alarm: both bands present together for enough frames
    if (n_co >= min_frames_alarm) and (frac_co >= min_frac_alarm) and (frac_aL > 0.05) and (frac_aH > 0.05):
        label, rule = "alarm", "alarm_co"

    # 2) High-only: high band dominates without alarm pattern
    elif (frac_high >= min_frac_high) and (frac_aH < 0.04 or frac_aL < 0.04 or frac_co < 0.03):
        label, rule = "high", "high_only"

    # 2b) NEW: near-tie favors HIGH (helps: high call + low noise)
    elif abs(frac_high - frac_low) <= near_tie_eps and frac_high >= 0.06:
        label, rule = "high", "high_near_tie"

    # 2c) NEW: short high streak lock-in
    elif high_streak >= min_high_streak:
        label, rule = "high", "high_streak"

    # 3) Otherwise: low if it clears the bar
    elif (frac_low >= min_frac_low):
        label, rule = "low", "low_frac"

    else:
        # fall back: pick the largest fraction among (low, alarm_co, high)
        which = int(np.argmax([frac_low, frac_co, frac_high]))
        label = ["low", "alarm", "high"][which]
        rule = "fallback_argmax"

    diag = dict(
        counts=dict(
            low=int(low_on.sum()),
            alarm_low=int(aL_on.sum()),
            alarm_high=int(aH_on.sum()),
            high=int(high_on.sum()),
            alarm_co=int(alarm_co.sum()),
            high_streak=high_streak,
            low_streak=low_streak,
            T=int(T),
        ),
        fracs=dict(low=frac_low, alarm_low=frac_aL, alarm_high=frac_aH,
                   high=frac_high, alarm_co=frac_co),
        thresholds=dict(
            min_frames_alarm=min_frames_alarm, min_frac_alarm=min_frac_alarm,
            min_frac_high=min_frac_high, min_frac_low=min_frac_low,
            near_tie_eps=near_tie_eps, min_high_streak=min_high_streak
        ),
        rule=rule
    )
    return label, diag

# def classify_spectrogram_three_way(S, freqs_khz, *,
#                                    low_band=(4, 14),          # gerbil low noise band
#                                    alarm_low=(18, 26),        # alarm fundamental
#                                    alarm_high=(40, 55),       # alarm 2nd harmonic
#                                    high_band=(28, 60),        # high-only calls
#                                    co_win=1,                  # allow ±1 frame for co-occ
#                                    min_frames_alarm=6,        # absolute frames
#                                    min_frac_alarm=0.06,       # fraction of T
#                                    min_frac_high=0.12,        # high-only fraction
#                                    min_frac_low=0.12,
#                                    near_tie_eps=0.03,  # if |frac_high-frac_low| ≤ eps → favor HIGH
#                                    min_high_streak=3  # lock HIGH if ≥ this many consecutive high frames
#                                    ):        # low fraction
#     """
#     Returns: label in {"low","alarm","high"} plus diagnostics.
#     """
#     F, T = S.shape
#     A = adaptive_activity_mask(S)  # boolean (F, T)
#
#     # band masks
#     low_idx   = band_idx(freqs_khz, *low_band)
#     aL_idx    = band_idx(freqs_khz, *alarm_low)
#     aH_idx    = band_idx(freqs_khz, *alarm_high)
#     high_idx  = band_idx(freqs_khz, *high_band)
#
#     # per-column hits
#     low_hits  = np.count_nonzero(A[low_idx], axis=0)    # (T,)
#     aL_hits   = np.count_nonzero(A[aL_idx], axis=0)
#     aH_hits   = np.count_nonzero(A[aH_idx], axis=0)
#     high_hits = np.count_nonzero(A[high_idx], axis=0)
#
#     # per-frame activity booleans - new
#     low_on = (low_hits > 0)
#     aL_on = (aL_hits > 0)
#     aH_on = (aH_hits > 0)
#     high_on = (high_hits > 0)
#
#
#     # co-occurrence: dilate in time so near-simultaneous counts match
#     if co_win > 0:
#         st = (1, 2*co_win+1)  # (freq,time)
#         aL_act = binary_dilation(A[aL_idx], structure=np.ones(st)).any(axis=0)
#         aH_act = binary_dilation(A[aH_idx], structure=np.ones(st)).any(axis=0)
#     else:
#         aL_act, aH_act = aL_on, aH_on
#
#     alarm_co = aL_act & aH_act
#
#     # fractions across time
#     def _frac(x):  # safe for T==0
#         return float(np.mean(x)) if T else 0.0
#
#     frac_low = _frac(low_on)
#     frac_aL = _frac(aL_on)
#     frac_aH = _frac(aH_on)
#     frac_high = _frac(high_on)
#     frac_co = _frac(alarm_co)
#
#     n_co = int(alarm_co.sum())
#
#     # --- Decision rules (ordered) ---
#     # 1) Alarm: both bands present together for enough frames
#     if (n_co >= min_frames_alarm) and (frac_co >= min_frac_alarm) and (frac_aL > 0.05) and (frac_aH > 0.05):
#         label = "alarm"
#     # 2) High-only: high band dominates without alarm pattern
#     elif (frac_high >= min_frac_high) and (frac_aH < 0.04 or frac_aL < 0.04 or frac_co < 0.03):
#         label = "high"
#     # 3) Otherwise: low
#     elif (frac_low >= min_frac_low):
#         label = "low"
#     else:
#         # fall back: pick the largest fraction
#         label = ["low","alarm","high"][np.argmax([frac_low, frac_co, frac_high])]
#
#     diag = dict(
#         frac=dict(low=frac_low, alarm_low=frac_aL, alarm_high=frac_aH, high=frac_high, alarm_co=frac_co),
#         frames=dict(T=int(T), alarm_co=n_co),
#         thresholds=dict(min_frames_alarm=min_frames_alarm, min_frac_alarm=min_frac_alarm,
#                         min_frac_high=min_frac_high, min_frac_low=min_frac_low),
#     )
#     return label, diag

# -----------
# -----------
# -----------

# save_as_bins.py
# ----------------
# Batch-load spectrograms, classify with your existing function,
# and save review PNGs into bin folders: low/, alarm/, high/.

# ======== IMPORT YOUR CLASSIFIER ========
# Make sure this import/path matches where you defined it.
# from your_module import classify_spectrogram_three_way

# For illustration here, I’ll assume the function is in scope:
# def classify_spectrogram_three_way(S, freqs_khz): ...


# ======== HELPERS ========

def ensure_dirs(root: Path):
    for name in ("low", "alarm", "high"):
        (root / name).mkdir(parents=True, exist_ok=True)

def to_db(X, eps=1e-12):
    X = np.asarray(X, float)
    return 20.0 * np.log10(np.maximum(X, eps))

def autoscale_limits(img_db):
    # robust color limits for nicer contrast in plots
    vmin = np.percentile(img_db, 5)
    vmax = np.percentile(img_db, 99)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = None, None
    return vmin, vmax

def plot_and_save(S, freqs_khz, out_png, title=None):
    S_db = to_db(S)
    vmin, vmax = autoscale_limits(S_db)

    F, T = S_db.shape
    extent = [0, T, freqs_khz.min(), freqs_khz.max()]  # x=frames, y=kHz

    plt.figure(figsize=(7.2, 5.4), dpi=FIG_DPI)
    im = plt.imshow(S_db, origin="lower", aspect="auto",
                    extent=extent, vmin=vmin, vmax=vmax, cmap="magma")
    plt.colorbar(im, label="dB")
    plt.xlabel("Time (frames)")
    plt.ylabel("Frequency (kHz)")
    if title:
        plt.title(title, fontsize=10)

    # Optional dashed band lines (for quick visual QA)
    for k, y in VIZ_BANDS.items():
        if y is not None and (freqs_khz.min() <= y <= freqs_khz.max()):
            plt.axhline(y, ls="--", lw=1, color="white", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


# ======== MAIN ========

def main():
    ensure_dirs(OUT)
    input_files = glob.glob(str(INPUT / "*.hdf5"))
    for input_file in tqdm(input_files):
        with h5py.File(input_file, "r") as f:
            Sds = f[DS_SPECS]  # (N,F,T) or (F,T)
            # freqs
            # if DS_FREQS and DS_FREQS in f:
            #     freqs_khz = np.array(f[DS_FREQS])  # (F,)
            # elif DS_FREQS_HZ and DS_FREQS_HZ in f:
            #     freqs_khz = np.array(f[DS_FREQS_HZ]) / 1000.0
            # else:
            #     raise KeyError("Need freqs_khz or freqs_hz in the HDF5.")
            freqs_khz = FREQ_AXIS_KHZ
            # unify to iterable over samples
            if Sds.ndim == 2:
                iterable = [(0, np.array(Sds))]
            elif Sds.ndim == 3:
                n = Sds.shape[0]
                if MAX_ITEMS is not None:
                    n = min(n, int(MAX_ITEMS))
                iterable = ((i, Sds[i]) for i in range(n))
            else:
                raise ValueError(f"Unsupported spectrogram shape: {Sds.shape}")

            for i, S in tqdm(iterable):
                S = np.array(S)  # force load from h5py
                # call YOUR classifier
                label, diag = classify_spectrogram_three_way(S, freqs_khz)

                # build filename with a bit of diag for later inspection
                frac = diag.get("frac", {})
                frac_str = f"L={frac.get('low',0):.2f}_Aco={frac.get('alarm_co',0):.2f}_H={frac.get('high',0):.2f}"
                out_png = OUT / label / f"item_{i:06d}__{frac_str}.png"

                f = diag["fracs"];
                r = diag["rule"]
                title = f'{label.upper()} | L={f["low"]:.2f}_Aco={f["alarm_co"]:.2f}_H={f["high"]:.2f}  ({r})'
                plot_and_save(S, freqs_khz, out_png, title=title)

        print(f"Done. Review results in:\n  {OUT / 'low'}\n  {OUT / 'alarm'}\n  {OUT / 'high'}")

if __name__ == "__main__":
    main()
