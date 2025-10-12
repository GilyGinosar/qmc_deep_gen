"""
Export per-bin spectrogram PNGs (with mean-frequency line) and one "summed" spectrogram per bin.

Inputs (via CSV):
- One row per item/call.
- Required columns:
    - 'bin'               : integer (or str) bin id for the item
    - 'mean_freq_hz'      : mean frequency in Hz for the item
- And EITHER:
    - 'spec_path'         : path to a saved numpy .npy spectrogram (2D array [freq_bins x time_frames])
      (OPTIONAL per-row 'freq_max_hz' if your spec uses a known max frequency; otherwise use CONFIG.freq_max_hz)
  OR
    - 'audio_path'        : path to an audio file; STFT will be computed
      (OPTIONAL per-row 'sr' for item-specific sampling rate; otherwise use CONFIG.sr)

Outputs:
- <OUTPUT_ROOT>/
    bin_<id>/
        <row_id>_<optional_name>.png           # each individual spectrogram, y-axis in kHz, mean-freq line
        SUM_spectrogram.png                     # summed spectrogram (same shape for all items in the bin)
"""

import os
import math
import json
import pathlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import zoom  # for resizing to a common shape

# --- Optional: only needed if you want the script to compute spectrograms from audio paths ---
try:
    import librosa
except Exception:
    librosa = None  # We'll guard usage with a clear error if needed.


# ======================
# CONFIGURATION
# ======================

@dataclass
class CONFIG:
    # --- Inputs ---
    csv_path: str = r"/path/to/your/metadata.csv"

    # Column names in the CSV
    col_bin: str = "bin"
    col_mean_freq_hz: str = "mean_freq_hz"
    col_spec_path: str = "spec_path"       # if using precomputed spectrograms (.npy)
    col_audio_path: str = "audio_path"     # if computing STFT (fallback when spec_path is missing)
    col_sr: str = "sr"                      # optional per-row sampling rate (Hz)
    col_id: str = "id"                      # optional identifier for nicer filenames
    col_label: str = "label"                # optional label to add to filename

    # --- Output root ---
    output_root: str = r"/path/to/output/folder"

    # --- Plot appearance ---
    cmap: str = "magma"                     # good for spectrograms
    dpi: int = 200
    individual_figsize: Tuple[float, float] = (5.0, 4.0)
    summed_figsize: Tuple[float, float] = (6.0, 4.5)

    # --- Spectrogram assumptions if loading .npy ---
    # If you load precomputed spectrograms from .npy and you DON'T know the true Hz scale,
    # we will render y-axis 0..freq_max_hz (in kHz). Set this to your known Nyquist or band ceiling.
    freq_max_hz_for_precomputed: float = 48000.0  # change to match your preprocessing pipeline

    # --- STFT settings (only used if computing from audio) ---
    sr_default: int = 96000           # used if you don't have per-row 'sr'
    n_fft: int = 1024
    hop_length: int = 256
    window: str = "hann"
    use_power_db: bool = True         # convert |STFT|^2 to dB scale for plotting
    top_db: float = 80.0              # dynamic range if converting to dB

    # --- Resizing for SUM ---
    # All specs in a bin will be resized to this shape before summing.
    # If None, the script will use the MAX (freq_bins, time_bins) seen in that bin.
    target_sum_shape: Optional[Tuple[int, int]] = None

    # --- Normalization before summing (helps avoid one item dominating) ---
    per_item_normalize: bool = True   # normalize each spectrogram by its max before summing

CFG = CONFIG()


# ======================
# HELPERS
# ======================

def ensure_dir(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def compute_stft_spectrogram(
    audio_path: str,
    sr: int,
    n_fft: int,
    hop_length: int,
    window: str,
    as_db: bool,
    top_db: float
) -> Tuple[np.ndarray, float]:
    """
    Returns (spec_2d, freq_max_hz). spec_2d shape: [freq_bins, time_frames]
    """
    if librosa is None:
        raise RuntimeError(
            "librosa is not installed, but it's required to compute spectrograms from audio. "
            "Install with `pip install librosa` or provide precomputed .npy spectrograms."
        )
    y, sr_loaded = librosa.load(audio_path, sr=sr)  # resample (or exact if file already at sr)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window, center=True)
    S_mag = np.abs(S)

    # Power or magnitude in dB for plotting clarity:
    if as_db:
        S_pow = S_mag**2
        spec = librosa.power_to_db(S_pow, ref=np.max, top_db=top_db)
    else:
        spec = S_mag

    freq_max_hz = sr_loaded / 2.0
    return spec, freq_max_hz


def load_precomputed_spec(spec_path: str) -> np.ndarray:
    """
    Load a precomputed 2D spectrogram (freq x time) from .npy
    """
    arr = np.load(spec_path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array at {spec_path}, got shape {arr.shape}")
    return arr


def resize_spec(spec: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resize a 2D array to target_shape using scipy.ndimage.zoom (bilinear-like interpolation).
    """
    f, t = spec.shape
    f_tgt, t_tgt = target_shape
    zoom_f = f_tgt / max(f, 1)
    zoom_t = t_tgt / max(t, 1)
    return zoom(spec, (zoom_f, zoom_t), order=1)  # order=1 ~ bilinear


def plot_spec_to_png(
    spec: np.ndarray,
    out_path: str,
    mean_freq_hz: float,
    freq_max_hz: float,
    cmap: str,
    figsize: Tuple[float, float],
    dpi: int
):
    """
    Plot a spectrogram with y-axis in kHz and a horizontal line at mean_freq_hz.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    # extent: x from 0..T, y from 0..freq_max_hz_kHz
    freq_max_khz = freq_max_hz / 1000.0
    extent = [0, spec.shape[1], 0.0, freq_max_khz]

    # For better visual contrast, auto vmin/vmax from percentiles (works for linear or dB):
    vmin = np.percentile(spec, 5)
    vmax = np.percentile(spec, 95)
    if math.isclose(vmin, vmax):
        vmin, vmax = spec.min(), spec.max()

    im = plt.imshow(
        spec,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    plt.colorbar(im, pad=0.01, shrink=0.9)

    # Mean frequency line in kHz:
    mean_khz = mean_freq_hz / 1000.0
    plt.axhline(mean_khz, color="white", linestyle="--", linewidth=1.0, alpha=0.9)

    plt.xlabel("Time (frames)")
    plt.ylabel("Frequency (kHz)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_sum_spec_to_png(
    spec_sum: np.ndarray,
    out_path: str,
    freq_max_hz: float,
    cmap: str,
    figsize: Tuple[float, float],
    dpi: int
):
    """
    Plot the summed spectrogram for a bin, with y-axis in kHz.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    freq_max_khz = freq_max_hz / 1000.0
    extent = [0, spec_sum.shape[1], 0.0, freq_max_khz]

    vmin = np.percentile(spec_sum, 5)
    vmax = np.percentile(spec_sum, 95)
    if math.isclose(vmin, vmax):
        vmin, vmax = spec_sum.min(), spec_sum.max()

    im = plt.imshow(
        spec_sum,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    plt.colorbar(im, pad=0.01, shrink=0.9)
    plt.xlabel("Time (frames)")
    plt.ylabel("Frequency (kHz)")
    plt.title("Summed spectrogram")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def safe_filename(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))


# ======================
# MAIN
# ======================

def main():
    CFG.output_root and ensure_dir(CFG.output_root)
    df = pd.read_csv(CFG.csv_path)

    # Sanity: check required columns
    if CFG.col_bin not in df.columns or CFG.col_mean_freq_hz not in df.columns:
        raise ValueError(
            f"CSV must contain '{CFG.col_bin}' and '{CFG.col_mean_freq_hz}' columns."
        )
    if (CFG.col_spec_path not in df.columns) and (CFG.col_audio_path not in df.columns):
        raise ValueError(
            f"CSV must contain either '{CFG.col_spec_path}' (paths to .npy spectrograms) "
            f"or '{CFG.col_audio_path}' (paths to audio files)."
        )

    # Group rows by bin
    grouped = df.groupby(CFG.col_bin)

    for bin_id, g in grouped:
        bin_dir = os.path.join(CFG.output_root, f"bin_{safe_filename(bin_id)}")
        ensure_dir(bin_dir)

        # Decide the target shape for summing in this bin
        # If not provided, pick max(F, T) across items in the bin
        shapes = []
        per_item_info = []  # list of (spec, mean_freq_hz, freq_max_hz, suggested_name)
        for idx, row in g.iterrows():
            mean_freq_hz = float(row[CFG.col_mean_freq_hz])

            # 1) Load or compute spectrogram
            if CFG.col_spec_path in row and isinstance(row[CFG.col_spec_path], str) and os.path.isfile(row[CFG.col_spec_path]):
                spec = load_precomputed_spec(row[CFG.col_spec_path])
                # freq_max_hz for precomputed arrays is assumed (unless provided as per-row column)
                freq_max_hz = float(row.get("freq_max_hz", CFG.freq_max_hz_for_precomputed))
            else:
                if CFG.col_audio_path not in row or not isinstance(row[CFG.col_audio_path], str):
                    print(f"Skipping row {idx}: no valid spectrogram or audio path.")
                    continue
                sr = int(row.get(CFG.col_sr, CFG.sr_default))
                spec, freq_max_hz = compute_stft_spectrogram(
                    row[CFG.col_audio_path], sr,
                    CFG.n_fft, CFG.hop_length, CFG.window,
                    as_db=CFG.use_power_db, top_db=CFG.top_db
                )

            shapes.append(spec.shape)

            # Build a nice filename stem for the individual image
            stem_parts = []
            if CFG.col_id in row and not pd.isna(row[CFG.col_id]):
                stem_parts.append(str(row[CFG.col_id]))
            if CFG.col_label in row and not pd.isna(row[CFG.col_label]):
                stem_parts.append(str(row[CFG.col_label]))
            if CFG.col_spec_path in row and isinstance(row[CFG.col_spec_path], str):
                stem_parts.append(pathlib.Path(row[CFG.col_spec_path]).stem)
            elif CFG.col_audio_path in row and isinstance(row[CFG.col_audio_path], str):
                stem_parts.append(pathlib.Path(row[CFG.col_audio_path]).stem)

            stem = "_".join(p for p in stem_parts if p) or f"row{idx}"
            stem = safe_filename(stem)

            per_item_info.append((spec, mean_freq_hz, freq_max_hz, stem))

        if not per_item_info:
            print(f"[bin {bin_id}] No valid items found, skipping.")
            continue

        # Determine target sum shape
        if CFG.target_sum_shape is not None:
            tgt_f, tgt_t = CFG.target_sum_shape
        else:
            # Use the largest F and largest T found in this bin
            tgt_f = max(s[0] for s in shapes)
            tgt_t = max(s[1] for s in shapes)
        target_shape = (tgt_f, tgt_t)

        # For summed spectrogram we need a consistent freq_max_hz (y-scale).
        # Best practice: if STFT route, this is Nyquist; if precomputed, use your configured ceiling.
        # Here we choose the MAX freq_max_hz across items so the y-axis is a safe upper bound.
        freq_max_hz_for_bin = max(info[2] for info in per_item_info)

        # Start the sum
        spec_sum = np.zeros(target_shape, dtype=np.float32)

        # Export individual PNGs, accumulate the sum (with optional per-item normalization)
        for spec, mean_freq_hz, freq_max_hz, stem in per_item_info:
            # Resize to target shape for summing
            spec_resized = resize_spec(spec, target_shape)

            if CFG.per_item_normalize:
                m = np.max(spec_resized)
                if m > 0:
                    spec_resized = spec_resized / m

            spec_sum += spec_resized

            # Plot & save the individual image using the original freq_max_hz for correct y-scale
            out_png = os.path.join(bin_dir, f"{stem}.png")
            plot_spec_to_png(
                spec=spec,
                out_path=out_png,
                mean_freq_hz=mean_freq_hz,
                freq_max_hz=freq_max_hz,
                cmap=CFG.cmap,
                figsize=CFG.individual_figsize,
                dpi=CFG.dpi
            )

        # Save the summed spectrogram PNG (with the bin's chosen freq_max_hz)
        out_sum_png = os.path.join(bin_dir, "SUM_spectrogram.png")
        plot_sum_spec_to_png(
            spec_sum=spec_sum,
            out_path=out_sum_png,
            freq_max_hz=freq_max_hz_for_bin,
            cmap=CFG.cmap,
            figsize=CFG.summed_figsize,
            dpi=CFG.dpi
        )

        # Optionally, also save the raw sum array (useful for later)
        np.save(os.path.join(bin_dir, "SUM_spectrogram.npy"), spec_sum)

        print(f"[bin {bin_id}] Wrote {len(per_item_info)} images + SUM to: {bin_dir}")


if __name__ == "__main__":
    main()
