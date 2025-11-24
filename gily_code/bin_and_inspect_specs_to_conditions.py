"""
Unified driver for spectrogram binning/preview.
It wraps the two existing flows without modifying them:
  - count mode (uses bin_specs.count_all_bins_decision)
  - rule mode  (uses bin_specs_rule_based.classify_spectrogram_three_way)

Select mode via --mode {count,rule}. All configs are in the CONFIG section.
Existing scripts remain untouched; this file just reuses their helpers.


 - Count mode (--mode count): Three bins (low/mid/high). It thresholds the spectrogram, counts active pixels in three
    frequency bands, and picks the bin with a mid-priority rule and tie margins. Output: per-item PNGs and summed
    spectrograms per bin.
  - Rule mode (--mode rule): Three classes (low/alarm/high). It uses a more involved heuristic: adaptive per-column
    activity masks, band co-occurrence for alarm (fundamental + harmonic), streak and near-tie logic to favor high, and
    fractions over time. Output: per-item PNGs labeled low/alarm/high with diagnostic info in filenames.

  In short: count mode is a simple band-counting low/mid/high sorter; rule mode is a richer detector distinguishing
  alarm vs high vs low.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

import gily_code.all.bin_specs as bs
import gily_code.all.bin_specs_rule_based as brb


# -----------------------
# CONFIG (edit to taste)
# -----------------------
class CONFIG:
    # Shared
    roots_per_family = {
        1: [r"D:\Data\237"],
        # 2: [...]
    }
    families = [1]
    specs_dataset = "specs"  # dataset name inside HDF5
    max_items = None  # limit per file for quick tests; None for all

    # Count mode settings (3 bins: low/mid/high)
    count_out_dir = r"D:\Data\Figs_fastdump_counts_unified"
    freq_min_hz = 500.0
    freq_max_hz = 62500.0
    num_freq_bins = 128
    edges_hz = (18_000.0, 26_000.0)  # low < e1, mid [e1,e2), high >= e2
    noise_floor_percentile = 70.0
    min_frame_active_ratio = 0.01
    min_active_pixels = 50
    mid_priority_frac = 0.25
    tie_margin = 0.02

    # Rule mode settings (low/alarm/high)
    rule_out_dir = r"D:\Data\bin_previews_rule_unified"
    freq_axis_khz = np.linspace(freq_min_hz / 1000.0, freq_max_hz / 1000.0, num_freq_bins)
    # brb classifier band defaults are used; adjust inside brb.classify_spectrogram_three_way if needed.


# -----------------------
# Helpers
# -----------------------
def list_family_files(roots_per_family: dict, families: list[int]) -> list[str]:
    """Collect *.hdf5 under {root}/processed-data/family{fam} for all families."""
    files: list[str] = []
    for fam in families:
        roots = roots_per_family.get(fam, [])
        for root in roots:
            spec_dir = os.path.join(root, "processed-data", f"family{fam}")
            files.extend(Path(spec_dir).glob("*.hdf5"))
    return sorted({str(p) for p in files})


# -----------------------
# Count mode
# -----------------------
def run_count_mode(cfg: CONFIG):
    Path(cfg.count_out_dir).mkdir(parents=True, exist_ok=True)
    files = list_family_files(cfg.roots_per_family, cfg.families)
    if not files:
        print("[count] no files found")
        return

    # Prepare bin dirs and accumulators
    bin_dirs = {
        0: os.path.join(cfg.count_out_dir, "bin_0_low"),
        1: os.path.join(cfg.count_out_dir, "bin_1_mid"),
        2: os.path.join(cfg.count_out_dir, "bin_2_high"),
    }
    for d in bin_dirs.values():
        os.makedirs(d, exist_ok=True)

    sum_accum = {0: None, 1: None, 2: None}
    sum_shape = {0: None, 1: None, 2: None}
    running_idx = {0: 0, 1: 0, 2: 0}

    freq_axis_hz = np.linspace(cfg.freq_min_hz, cfg.freq_max_hz, cfg.num_freq_bins, dtype=np.float64)

    print(f"[count] processing {len(files)} files...")
    for fn in tqdm(files):
        with h5py.File(fn, "r", locking=False) as f:
            specs = f[cfg.specs_dataset]
            N = len(specs) if cfg.max_items is None else min(len(specs), int(cfg.max_items))
            for i in range(N):
                spec = np.array(specs[i], dtype=np.float64)

                counts, fracs, winner, rule = bs.count_all_bins_decision(
                    spec,
                    freq_axis_hz=freq_axis_hz,
                    edges_hz=cfg.edges_hz,
                    noise_floor_percentile=cfg.noise_floor_percentile,
                    min_frame_active_ratio=cfg.min_frame_active_ratio,
                    min_active_pixels=cfg.min_active_pixels,
                    mid_priority_frac=cfg.mid_priority_frac,
                    tie_margin=cfg.tie_margin,
                )

                b = int(winner)
                cL, cM, cH = (int(counts[0]), int(counts[1]), int(counts[2]))
                fL, fM, fH = (float(fracs[0]), float(fracs[1]), float(fracs[2]))
                subtitle = f"L/M/H={cL}/{cM}/{cH}  frac=[{fL:.2f},{fM:.2f},{fH:.2f}]  -> bin {b} ({rule})"

                out_png = os.path.join(
                    bin_dirs[b],
                    bs.safe_name(f"{Path(fn).stem}_idx{i:05d}.png"),
                )
                bs.save_spec_png(
                    spec,
                    out_png,
                    freq_axis_hz=freq_axis_hz,
                    edges_hz=cfg.edges_hz,
                    title=None,
                    subtitle=subtitle,
                    cmap=bs.CMAP,
                    dpi=bs.DPI_INDIV,
                    figsize=bs.FIGSIZE_INDIV,
                )

                # Sum image (per bin), normalized per item
                tgt = sum_shape[b]
                if tgt is None:
                    tgt = spec.shape
                    sum_shape[b] = tgt
                    sum_accum[b] = np.zeros(tgt, dtype=np.float32)

                r = bs.resize_to(spec, tgt).astype(np.float32)
                mx = float(r.max())
                if mx > 0:
                    r = r / mx
                sum_accum[b] += r
                running_idx[b] += 1

    # Save sums
    for b in (0, 1, 2):
        if sum_accum[b] is None:
            print(f"[count] bin {b} empty; skipping SUM")
            continue
        sum_png = os.path.join(bin_dirs[b], "SUM_spectrogram.png")
        bs.save_spec_png(
            sum_accum[b],
            sum_png,
            freq_axis_hz=freq_axis_hz,
            edges_hz=cfg.edges_hz,
            title=f"SUM bin {b}",
            subtitle=None,
            cmap=bs.CMAP,
            dpi=bs.DPI_SUM,
            figsize=bs.FIGSIZE_SUM,
        )
        np.save(os.path.join(bin_dirs[b], "SUM_spectrogram.npy"), sum_accum[b])
        print(f"[count] bin {b}: saved SUM to {sum_png} (count={running_idx[b]})")


# -----------------------
# Rule mode
# -----------------------
def run_rule_mode(cfg: CONFIG):
    out_root = Path(cfg.rule_out_dir)
    brb.ensure_dirs(out_root)
    files = list_family_files(cfg.roots_per_family, cfg.families)
    if not files:
        print("[rule] no files found")
        return

    print(f"[rule] processing {len(files)} files...")
    for fn in tqdm(files):
        with h5py.File(fn, "r") as f:
            Sds = f[cfg.specs_dataset]
            n = Sds.shape[0] if Sds.ndim == 3 else 1
            if cfg.max_items is not None:
                n = min(n, int(cfg.max_items))
            iterable = [(i, Sds[i]) for i in range(n)] if Sds.ndim == 3 else [(0, Sds[:])]

            for i, S in tqdm(iterable):
                S = np.array(S)
                label, diag = brb.classify_spectrogram_three_way(S, cfg.freq_axis_khz)

                frac = diag.get("fracs", {})
                frac_str = f"L={frac.get('low',0):.2f}_Aco={frac.get('alarm_co',0):.2f}_H={frac.get('high',0):.2f}"
                out_png = out_root / label / f"{Path(fn).stem}_idx{i:05d}__{frac_str}.png"

                fvals = diag["fracs"]
                r = diag["rule"]
                title = f'{label.upper()} | L={fvals["low"]:.2f}_Aco={fvals["alarm_co"]:.2f}_H={fvals["high"]:.2f}  ({r})'
                brb.plot_and_save(S, cfg.freq_axis_khz, out_png, title=title)

    print(f"[rule] done. See {out_root}")


# -----------------------
# Entry
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Unified binning driver")
    parser.add_argument("--mode", choices=["count", "rule"], required=True, help="Which binning flow to run")
    args = parser.parse_args()

    cfg = CONFIG()
    if args.mode == "count":
        run_count_mode(cfg)
    else:
        run_rule_mode(cfg)


if __name__ == "__main__":
    main()
