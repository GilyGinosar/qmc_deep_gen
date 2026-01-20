import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from skimage import restoration, filters, morphology, feature
from pathlib import Path
import pandas as pd

# --- 1. CONFIGURATION & PATHS ---
exp = 235
target_channel = 20
data_path = fr"\\sanesstorage.cns.nyu.edu\archive\ginosar\Processed_data\Audio\{exp}"
wavpath = os.path.join(data_path, 'Averaged_wavs_w_annotations')
onoffpath = os.path.join(data_path, 'vox_for_Miles.csv')
specpath = os.path.join(data_path, 'processed-data/family1_FinalTuned_v2')
Path(specpath).mkdir(parents=True, exist_ok=True)

# Parameters for Context
CONTEXT_SEC = 2.0  # Total window size to help the algorithm 'see' noise
PADDING_SEC = 0.2  # Final visual padding for the call

# --- 2. PREPARE DATA ---
voc_data = pd.read_csv(onoffpath)
# Focus on Channel 30
voc_data = voc_data[voc_data['assigned_channel'] == target_channel].copy()
voc_data['full_path'] = voc_data.apply(
    lambda r: os.path.join(wavpath, f"channel_{r.assigned_channel}_file_{str(r.file_num).zfill(3)}.wav"), axis=1
)


def apply_zonal_cleaning(img, sample_rate):
    """
    Advanced gradient-aware cleaning to preserve edgy calls but kill smeared noise.
    """
    # 1. Denoise gently
    denoised = restoration.denoise_tv_chambolle(img, weight=0.08)

    # 2. Multi-scale Ridges (Catches both thin ticks and thick harmonics)
    ridges = filters.meijering(denoised, sigmas=[1, 2], black_ridges=False)
    ridge_binary = ridges > filters.threshold_otsu(ridges)

    # 3. Canny Edges (Captures local contrast boundaries)
    edges = feature.canny(denoised, sigma=1.0, low_threshold=0.1, high_threshold=0.2)

    # 4. Zonal Thresholding
    strict_thresh = filters.threshold_local(denoised, block_size=51, offset=0.08)
    binary_strict = denoised > strict_thresh

    rows, cols = binary_strict.shape
    ten_khz_row = int((10000 / (sample_rate / 2)) * rows)
    skeleton = edges | ridge_binary  # Use edges or ridges to find the call structure

    # --- ZONE 1: Low Frequency (Stacks & Foot Thumps) ---
    low_zone = (binary_strict & skeleton)  # Must be edgy AND bright
    low_zone[ten_khz_row:, :] = 0
    clean_low = morphology.binary_opening(low_zone, morphology.rectangle(1, 35)) | \
                morphology.binary_opening(low_zone, morphology.rectangle(15, 1))

    # --- ZONE 2: High Frequency (UFMs, Warbles, Alarms) ---
    high_zone = skeleton.copy()
    high_zone[:ten_khz_row, :] = 0
    clean_high = morphology.binary_opening(high_zone, morphology.rectangle(1, 8)) | \
                 morphology.binary_opening(high_zone, morphology.rectangle(8, 1))

    return morphology.remove_small_objects(clean_low | clean_high, min_size=50)


# --- 3. EXECUTION ---
for file_path, calls in voc_data.groupby('full_path'):
    if not os.path.exists(file_path): continue

    sample_rate, data = wavfile.read(file_path)
    if len(data.shape) > 1: data = data.mean(axis=1)
    file_folder = os.path.join(specpath, os.path.basename(file_path).replace('.wav', ''))
    Path(file_folder).mkdir(parents=True, exist_ok=True)

    for idx, call in calls.iterrows():
        # Context window centering
        mid_point = (call.start_time_file_sec + call.stop_time_file_sec) / 2
        c_start = max(0, mid_point - (CONTEXT_SEC / 2))
        c_stop = min(len(data) / sample_rate, mid_point + (CONTEXT_SEC / 2))

        audio_context = data[int(c_start * sample_rate): int(c_stop * sample_rate)]
        if len(audio_context) < 1024: continue

        # Generate Context-Aware Spectrogram
        power, freqs, bins, im = plt.specgram(audio_context, NFFT=512, Fs=sample_rate, noverlap=256)
        plt.close()

        spec_db = 10 * np.log10(power + 1e-10)
        spec_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())

        # Apply the gradient-aware cleaning
        full_mask = apply_zonal_cleaning(spec_norm, sample_rate)

        # Crop back to call + padding
        call_rel_start = call.start_time_file_sec - c_start - PADDING_SEC
        call_rel_stop = call.stop_time_file_sec - c_start + PADDING_SEC
        col_idx = np.where((bins >= call_rel_start) & (bins <= call_rel_stop))[0]
        if len(col_idx) == 0: continue

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        axes[0].imshow(spec_norm[:, col_idx], origin='lower', aspect='auto',
                       extent=[bins[col_idx[0]], bins[col_idx[-1]], 500, 62500], cmap='magma')
        axes[0].set_title(f"Call {idx} | Ch30 | Time: {call.start_time_file_sec:.2f}s")

        axes[1].imshow(full_mask[:, col_idx], origin='lower', aspect='auto',
                       extent=[bins[col_idx[0]], bins[col_idx[-1]], 500, 62500], cmap='gray')

        plt.tight_layout()
        plt.savefig(f"{file_folder}/call_{idx}_{call.start_time_file_sec:.2f}s.png")
        plt.close()

print(f"Finished processing Channel {target_channel}. Results in {specpath}")



# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import wavfile
# from skimage import restoration, filters, morphology, feature
# from pathlib import Path
# import pandas as pd
#
# # --- 1. CONFIGURATION & CHANNEL FILTERING ---
# exp = 235
# target_channel = 30  # Focusing on Channel 30 as requested
# data_path = fr"\\sanesstorage.cns.nyu.edu\archive\ginosar\Processed_data\Audio\{exp}"
# wavpath = os.path.join(data_path, 'Averaged_wavs_w_annotations')
# onoffpath = os.path.join(data_path, 'vox_for_Miles.csv')
# specpath = os.path.join(data_path, 'processed-data/family1_FinalTuned')
# Path(specpath).mkdir(parents=True, exist_ok=True)
#
# # Load and Filter CSV immediately
# voc_data = pd.read_csv(onoffpath)
# voc_data = voc_data[voc_data['assigned_channel'] == target_channel].copy()
# voc_data['full_path'] = voc_data.apply(
#     lambda r: os.path.join(wavpath, f"channel_{r.assigned_channel}_file_{str(r.file_num).zfill(3)}.wav"), axis=1
# )
#
#
# def apply_zonal_cleaning(img, sample_rate):
#     # Denoise gently to preserve faint UFMs
#     denoised = restoration.denoise_tv_chambolle(img, weight=0.08)
#
#     # 1. MULTI-SCALE RIDGE DETECTION (The UFM/Stack Specialist)
#     # sigmas=[1, 2] catches both thin ticks and thick harmonics
#     ridges = filters.meijering(denoised, sigmas=[1, 2], black_ridges=False)
#     ridge_binary = ridges > filters.threshold_otsu(ridges)
#
#     # 2. CANNY EDGES for Local Contrast
#     edges = feature.canny(denoised, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
#
#     # 3. ZONAL THRESHOLDING
#     strict_thresh = filters.threshold_local(denoised, block_size=51, offset=0.08)
#     binary_strict = denoised > strict_thresh
#
#     rows, cols = binary_strict.shape
#     ten_khz_row = int((10000 / (sample_rate / 2)) * rows)
#
#     # Combined skeleton of the call
#     skeleton = edges | ridge_binary
#
#     # --- ZONE 1: Low Frequency (Stacks & Thumps) ---
#     low_zone = (binary_strict & skeleton)
#     low_zone[ten_khz_row:, :] = 0
#     clean_low = morphology.binary_opening(low_zone, morphology.rectangle(1, 35)) | \
#                 morphology.binary_opening(low_zone, morphology.rectangle(15, 1))
#
#     # --- ZONE 2: High Frequency (UFMs, Warbles, Alarms) ---
#     high_zone = skeleton.copy()
#     high_zone[:ten_khz_row, :] = 0
#     clean_high = morphology.binary_opening(high_zone, morphology.rectangle(1, 8)) | \
#                  morphology.binary_opening(high_zone, morphology.rectangle(8, 1))
#
#     # 4. FINAL CLEANUP
#     return morphology.remove_small_objects(clean_low | clean_high, min_size=50)
#
#
# # --- 2. EXECUTION LOOP ---
# # ... (Use the same groupby loop from previous message to process unique files)
#
#
#
#
# # import os
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.io import wavfile
# # from skimage import restoration, filters, morphology, feature
# # from pathlib import Path
# # import pandas as pd
#
# # # --- CONFIGURATION ---
# # exp = 235
# # data_path = fr"\\sanesstorage.cns.nyu.edu\archive\ginosar\Processed_data\Audio\{exp}"
# # wavpath = os.path.join(data_path, 'Averaged_wavs_w_annotations')
# # onoffpath = os.path.join(data_path, 'vox_for_Miles.csv')
# # specpath = os.path.join(data_path, 'processed-data/family1_ContextCleaning')
# # Path(specpath).mkdir(parents=True, exist_ok=True)
# #
# # Parameters for Context
# CONTEXT_SEC = 2.0  # Total window size to help the algorithm 'see' noise
# PADDING_SEC = 0.2  # Final visual padding for the call
# #
# #
# # def apply_zonal_cleaning(img, sample_rate):
# #     # 1. Total Variation Denoising: Flattens the 'smeared' haze
# #     denoised = restoration.denoise_tv_chambolle(img, weight=0.1)
# #
# #     # 2. EDGE DETECTION (Canny): The 'Smear' Killer
# #     # sigma=1 finds sharp edges; it will ignore low-gradient smears
# #     edges = feature.canny(denoised, sigma=1.0)
# #
# #     # 3. RIDGE DETECTION: The UFM Specialist
# #     ridges = filters.meijering(denoised, sigmas=[1], black_ridges=False)
# #     ridge_binary = ridges > filters.threshold_otsu(ridges)
# #
# #     # 4. TRADITIONAL THRESHOLDING (Zonal)
# #     strict_thresh = filters.threshold_local(denoised, block_size=51, offset=0.08)
# #     binary_strict = denoised > strict_thresh
# #
# #     rows, cols = binary_strict.shape
# #     ten_khz_row = int((10000 / (sample_rate / 2)) * rows)  # 10kHz split
# #
# #     # --- COMBINE STRATEGIES ---
# #     # We use (Edges OR Ridges) to find the 'skeleton' of the call
# #     skeleton = edges | ridge_binary
# #
# #     # --- ZONE 1: Low Frequency (Stacks & Thumps) ---
# #     low_zone = (binary_strict & skeleton)  # Only keep edges that are also 'bright'
# #     low_zone[ten_khz_row:, :] = 0
# #     # Strict horizontal check to confirm it's a stable harmonic stack
# #     clean_low = morphology.binary_opening(low_zone, morphology.rectangle(1, 35)) | \
# #                 morphology.binary_opening(low_zone, morphology.rectangle(15, 1))
# #
# #     # --- ZONE 2: High Frequency (UFMs, Warbles, Alarms) ---
# #     high_zone = skeleton.copy()
# #     high_zone[:ten_khz_row, :] = 0
# #     # UFMs and Warbles are preserved here because they have sharp edges
# #     clean_high = (morphology.binary_opening(high_zone, morphology.rectangle(1, 10)) | \
# #                   morphology.binary_opening(high_zone, morphology.rectangle(10, 1)))
# #
# #     # 5. FINAL MORPHOLOGICAL CLEANUP
# #     return morphology.remove_small_objects(clean_low | clean_high, min_size=60)
# # --- EXECUTION ---
# voc_data = pd.read_csv(onoffpath)
# voc_data['full_path'] = voc_data.apply(
#     lambda r: os.path.join(wavpath, f"channel_{r.assigned_channel}_file_{str(r.file_num).zfill(3)}.wav"), axis=1
# )
# # --- FILTER BY CHANNEL ---
# # Change this to 10, 20, or [10, 30] as needed
# target_channel = 30
# voc_data = voc_data[voc_data['assigned_channel'] == target_channel].copy()
#
# print(f"Focused on Channel {target_channel}. Total calls to process: {len(voc_data)}")
#
#
# for file_path, calls in voc_data.groupby('full_path'):
#     if not os.path.exists(file_path): continue
#
#     sample_rate, data = wavfile.read(file_path)
#     if len(data.shape) > 1: data = data.mean(axis=1)
#     file_folder = os.path.join(specpath, os.path.basename(file_path).replace('.wav', ''))
#     Path(file_folder).mkdir(parents=True, exist_ok=True)
#
#     for idx, call in calls.iterrows():
#         # 1. CALCULATE CONTEXT WINDOW (2.0s)
#         mid_point = (call.start_time_file_sec + call.stop_time_file_sec) / 2
#         c_start = max(0, mid_point - (CONTEXT_SEC / 2))
#         c_stop = min(len(data) / sample_rate, mid_point + (CONTEXT_SEC / 2))
#
#         audio_context = data[int(c_start * sample_rate): int(c_stop * sample_rate)]
#
#         # 2. GENERATE SPECTROGRAM ON FULL CONTEXT
#         # This gives the algorithms a large 'buffer' of noise to look at
#         power, freqs, bins, im = plt.specgram(audio_context, NFFT=512, Fs=sample_rate, noverlap=256)
#         plt.close()  # Don't show the full 2s plot yet
#
#         spec_db = 10 * np.log10(power + 1e-10)
#         spec_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())
#
#         # 3. APPLY CLEANING TO THE FULL CONTEXT
#         full_mask = apply_zonal_cleaning(spec_norm, sample_rate)
#
#         # 4. CROP RESULT BACK TO CALL + PADDING
#         # Find which columns in the 'bins' array correspond to the call time
#         call_rel_start = call.start_time_file_sec - c_start - PADDING_SEC
#         call_rel_stop = call.stop_time_file_sec - c_start + PADDING_SEC
#
#         col_idx = np.where((bins >= call_rel_start) & (bins <= call_rel_stop))[0]
#         if len(col_idx) == 0: continue
#
#         # Slicing the matrices for display
#         cropped_mask = full_mask[:, col_idx]
#         cropped_orig = spec_norm[:, col_idx]
#         cropped_bins = bins[col_idx]
#
#         # 5. PLOT VERTICAL STACKED INSPECTION
#         fig, axes = plt.subplots(2, 1, figsize=(12, 10))
#
#         # Top: Original (Cropped)
#         axes[0].imshow(cropped_orig, origin='lower', aspect='auto',
#                        extent=[cropped_bins[0], cropped_bins[-1], 500, 62500], cmap='magma')
#         axes[0].set_title(f"Call {idx} (Context-Aided Cleaning)")
#
#         # Bottom: Cleaned (Cropped)
#         axes[1].imshow(cropped_mask, origin='lower', aspect='auto',
#                        extent=[cropped_bins[0], cropped_bins[-1], 500, 62500], cmap='gray')
#
#         plt.tight_layout()
#         plt.savefig(f"{file_folder}/call_{idx}_{call.start_time_file_sec:.2f}s.png")
#         plt.close()
#
#
#
#
#
#
#
