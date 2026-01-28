import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from skimage import restoration, filters, morphology, feature
from pathlib import Path
import pandas as pd

# ==========================================
# 1. PATHS & CONFIGURATION
# ==========================================
EXP = 237
TARGET_CHANNEL = 30
DATA_PATH = fr"\\sanesstorage.cns.nyu.edu\archive\ginosar\Processed_data\Audio\{EXP}"
WAV_PATH = os.path.join(DATA_PATH, 'Averaged_wavs_w_annotations')
ON_OFF_PATH = os.path.join(DATA_PATH, 'vox_for_Miles.csv')
OUT_PATH = os.path.join(DATA_PATH, 'processed-data/family1_Cleaned')
Path(OUT_PATH).mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. PARAMETERS
# ==========================================
CONTEXT_SEC = 2.0
PADDING_SEC = 0.2
FREQ_SPLIT_HZ = 18000

TV_WEIGHT = 0.08
RIDGE_SIGMAS = [1, 2]
CANNY_SIGMA = 1.2
CANNY_low_th=0.1
CANNY_high_th=0.15 #0.2


STRICT_OFFSET = 0.08
NORMAL_OFFSET = 0.01 #0.03  # Lowered to protect faint UFM traces

BASE_H_LEN = 35
VOCAL_DISK_SIZE = 2
MIN_OBJ_SIZE = 60


# ==========================================
# 3. The cleaning function
# ==========================================
def apply_zonal_cleaning(img, sample_rate):
    denoised = restoration.denoise_tv_chambolle(img, weight=TV_WEIGHT)
    ridges = filters.meijering(denoised, sigmas=RIDGE_SIGMAS, black_ridges=False)
    ridge_binary = ridges > filters.threshold_otsu(ridges)
    edges = feature.canny(denoised, sigma=CANNY_SIGMA, low_threshold=CANNY_low_th, high_threshold=CANNY_high_th)
    skeleton = edges | ridge_binary

    rows, cols = skeleton.shape
    split_row = int((FREQ_SPLIT_HZ / (sample_rate / 2)) * rows)

    binary_strict = denoised > filters.threshold_local(denoised, block_size=51, offset=STRICT_OFFSET)
    binary_normal = denoised > filters.threshold_local(denoised, block_size=51, offset=NORMAL_OFFSET)

    # --- ZONE 1: Basement (supposed to pass stacks only) ---
    low_zone = (binary_strict & skeleton)
    low_zone[split_row:, :] = 0
    clean_low = morphology.binary_opening(low_zone, morphology.rectangle(1, BASE_H_LEN))

    # --- ZONE 2: Vocal Zone (safe for diagonal ufms and warbles) ---
    high_zone = (skeleton & binary_normal)
    high_zone[:split_row, :] = 0

    # Morphology.disk(2) provides multi-directional stability
    clean_high = morphology.binary_opening(high_zone, morphology.disk(VOCAL_DISK_SIZE))

    return morphology.remove_small_objects(clean_low | clean_high, min_size=MIN_OBJ_SIZE)


# ==========================================
# 4. EXECUTION
# ==========================================

if __name__ == "__main__":
    voc_data = pd.read_csv(ON_OFF_PATH)
    voc_data = voc_data[voc_data['assigned_channel'] == TARGET_CHANNEL].copy()
    voc_data['full_path'] = voc_data.apply(
        lambda r: os.path.join(WAV_PATH, f"channel_{r.assigned_channel}_file_{str(r.file_num).zfill(3)}.wav"), axis=1
    )

    for file_path, calls in voc_data.groupby('full_path'):
        if not os.path.exists(file_path): continue
        sample_rate, data = wavfile.read(file_path)
        print(sample_rate)
        if len(data.shape) > 1: data = data.mean(axis=1)
        file_folder = os.path.join(OUT_PATH, os.path.basename(file_path).replace('.wav', ''))
        Path(file_folder).mkdir(parents=True, exist_ok=True)

        for idx, call in calls.iterrows():
            mid_point = (call.start_time_file_sec + call.stop_time_file_sec) / 2
            c_start = max(0, mid_point - (CONTEXT_SEC / 2))
            c_stop = min(len(data) / sample_rate, mid_point + (CONTEXT_SEC / 2))
            audio_context = data[int(c_start * sample_rate): int(c_stop * sample_rate)]

            if len(audio_context) < 1024: continue
            power, freqs, bins, im = plt.specgram(audio_context, NFFT=512, Fs=sample_rate, noverlap=256)
            plt.close()

            spec_db = 10 * np.log10(power + 1e-10)
            spec_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())

            full_mask = apply_zonal_cleaning(spec_norm, sample_rate)

            call_rel_start = call.start_time_file_sec - c_start - PADDING_SEC
            call_rel_stop = call.stop_time_file_sec - c_start + PADDING_SEC
            col_idx = np.where((bins >= call_rel_start) & (bins <= call_rel_stop))[0]
            if len(col_idx) == 0: continue

            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            axes[0].imshow(spec_norm[:, col_idx], origin='lower', aspect='auto',
                           extent=[bins[col_idx[0]], bins[col_idx[-1]], 500, 62500], cmap='magma')
            axes[1].imshow(full_mask[:, col_idx], origin='lower', aspect='auto',
                           extent=[bins[col_idx[0]], bins[col_idx[-1]], 500, 62500], cmap='gray')
            plt.tight_layout()
            plt.savefig(f"{file_folder}/call_{idx}_{call.start_time_file_sec:.2f}s.png")
            plt.close()

print(f"Done. Check: {OUT_PATH}")

