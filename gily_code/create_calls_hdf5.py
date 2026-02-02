import pandas as pd
import numpy as np
from scipy.io import wavfile
from skimage.transform import resize
import os,glob
from tqdm import tqdm
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import inspect

from gily_code.ava_utils import get_spec
from ava.preprocessing.preprocess import get_syll_specs
from gily_code.clean_audio import apply_zonal_cleaning

### ---------------- Paths and parameters ----------------
exp = 237

data_path = fr"\\sanesstorage.cns.nyu.edu\archive\ginosar\Processed_data\Audio\{exp}" #'/mnt/home/mmartinez/ceph/data/gerbil/gily'
print(data_path)
onoffpath = os.path.join(data_path,'vox_for_Miles.csv')
wavpath = os.path.join(data_path,'Averaged_wavs_w_annotations') #'audio')
specpath = os.path.join(data_path,'processed-data/family1_TESTTEST')

Path(specpath).mkdir(parents=True, exist_ok=True)

params ={'get_spec': get_spec, # spectrogram maker,
'num_freq_bins':128,
'num_time_bins':128,
'min_freq': 500, # minimum frequency
'max_dur': 0.5, # maximum syllable duration'min_freq': 500, # minimum frequency (ralph used 0.3sec on warped specs), 1 sec made calls to small
'max_freq': 62500, # maximum frequency
'nperseg': 512, # FFT
'noverlap': 256, # FFT
'spec_min_val': -8, # minimum log-spectrogram value
'spec_max_val': -5, # maximum log-spectrogram value
'fs': 125000, # audio sample rate
'mel': False, # frequency spacing, mel or linear
'time_stretch': True, # stretch short syllables?
'within_syll_normalize': False, # normalize spectrogram values on a
                       # spectrogram-by-spectrogram basis
'max_num_syllables': None, # maximum number of syllables per directory
'sylls_per_file': 100, # syllable per file
'real_preprocess_params': ('min_freq', 'max_freq',
     'spec_min_val','spec_max_val', 'max_dur'), # tunable parameters
'int_preprocess_params': ('nperseg','noverlap'), # tunable parameters
'binary_preprocess_params': ('time_stretch', 'mel',
      'within_syll_normalize'), # tunable parameters
}

### ---------------- Prepare data lists ----------------
voc_data = pd.read_csv(onoffpath)

# channel_key = ['arena_1',
#               'arena_2',
#               'underground']
onsets,offsets = voc_data.start_time_file_sec,voc_data.stop_time_file_sec
lens = voc_data.duration_sec
channels = voc_data.assigned_channel
file_nums = voc_data.file_num
locs = voc_data.assigned_location

filenames = [os.path.join(wavpath,'channel_{}_file_{}.wav'.format(c,str(fn).zfill(3))) for c,fn in zip(channels,file_nums)]
print("Checking first filename:", filenames[0])
print("Does it exist?", os.path.exists(filenames[0]))

### ---------------- Main processing ----------------
# == Initialize
MASK_CONTEXT_SEC = 2.0  # Window used for mask calculation
SAVE_WINDOW_SEC = 0.3   # Final window saved to HDF5
BUFFER_SEC = 0.02    # 20ms padding around the actual call

write_file_num = 0
syll_data = {
    'specs':[], # masked (cleaned)
    'specs_raw':[], # unmasked
    'onsets':[],
    'offsets':[],
    'audio_filenames':[],
    'location':[]
}
sylls_per_file = params['sylls_per_file'] # CHNAGE THIS LATER
#threshold_db = -60


# == Main loop
for idx, (onset, offset, fn, loc) in enumerate(tqdm(zip(onsets, offsets, filenames, locs), total=len(filenames))):

    # -- 1. Gatekeeper: Skip Edge Calls
    midpoint = (onset + offset) / 2
    ml_start = midpoint - (SAVE_WINDOW_SEC / 2)
    ml_end = midpoint + (SAVE_WINDOW_SEC / 2)

    fs, audio = wavfile.read(fn)
    total_dur = len(audio) / fs
    if ml_start < 0 or ml_end > total_dur:
        continue  # Skip call if it's in the very edge - fix later

    # -- 2. Cleaning
    c_start = max(0, midpoint - (MASK_CONTEXT_SEC / 2))
    c_end = min(total_dur, midpoint + (MASK_CONTEXT_SEC / 2))
    audio_ctx = audio[int(c_start * fs): int(c_end * fs)]

    p_ctx, _, bins_ctx, _ = plt.specgram(audio_ctx, NFFT=512, Fs=fs, noverlap=256)
    plt.close()

    spec_db = 10 * np.log10(p_ctx + 1e-10)
    ctx_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())
    mask_ctx = apply_zonal_cleaning(ctx_norm, fs)

    # -- 3. Create the "Call-Only" Mask
    # Find indices in mask_ctx corresponding to onset-buffer and offset+buffer
    local_onset = (onset - c_start) - BUFFER_SEC
    local_offset = (offset - c_start) + BUFFER_SEC

    # Create a new blank mask for the call only
    call_only_mask = np.zeros_like(mask_ctx)
    call_cols = np.where((bins_ctx >= local_onset) & (bins_ctx <= local_offset))[0]
    call_only_mask[:, call_cols] = mask_ctx[:, call_cols]

    # -- 4. Extract ML Snippet & Apply Mask
    spec_list, valid = get_syll_specs([ml_start], [ml_end], fn, params)

    if valid:
        raw_spec = spec_list[0]

        # Crop the call-only mask to the 0.3s window
        actual_dur = c_end - c_start
        center_col = int(((midpoint - c_start) / actual_dur) * mask_ctx.shape[1])
        hw = int((SAVE_WINDOW_SEC / actual_dur) * (mask_ctx.shape[1] / 2))
        cropped_mask = call_only_mask[:, center_col - hw: center_col + hw]

        # Resize and multiply
        final_mask = resize(cropped_mask, (params['num_freq_bins'], params['num_time_bins']),
                            order=0, preserve_range=True, anti_aliasing=False)
        cleaned_spec = raw_spec * final_mask


        # Append to Buffer
        syll_data['specs'] += [cleaned_spec]
        syll_data['specs_raw'] += [raw_spec]
        syll_data['onsets'] += [onset]
        syll_data['offsets'] += [offset]
        syll_data['audio_filenames'] += [os.path.split(fn)[-1]]
        syll_data['location'] += [loc]

        # Run these in the Debug Console
        print(f"Context Cols: {mask_ctx.shape[1]}")
        print(f"Cropped Mask Cols: {cropped_mask.shape[1]}")
        print(f"Ratio: {cropped_mask.shape[1] / mask_ctx.shape[1]}")

        # -- 7. NEW: Comparison Plotting
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Get the filename for the title (e.g., channel_1_file_001.wav)
        base_fn = os.path.basename(fn)
        main_title = f"Exp: {exp} | File: {base_fn} | Call Index: {idx}"  #

        # Row 1: The Context Windows (2.0s)
        # Note: 'extent' is [left, right, bottom, top]
        ctx_extent = [0, MASK_CONTEXT_SEC, params['min_freq'], params['max_freq']]

        axes[0, 0].imshow(spec_db, origin='lower', aspect='auto', cmap='magma', extent=ctx_extent)
        axes[0, 0].set_title(f"{main_title}\nRaw spectrogram")
        axes[0, 0].set_ylabel("Frequency (Hz)")

        axes[1, 0].imshow(spec_db * mask_ctx, origin='lower', aspect='auto', cmap='magma', extent=ctx_extent)
        axes[1, 0].set_title("Mask")

        # Row 2: The Final HDF5 Slices (0.3s)
        # These are centered on the call midpoint
        save_extent = [0, SAVE_WINDOW_SEC, params['min_freq'], params['max_freq']]

        axes[0, 1].imshow(raw_spec, origin='lower', aspect='auto', cmap='magma', extent=save_extent)
        axes[0, 1].set_title(f"Final {SAVE_WINDOW_SEC}s Raw (HDF5)")
        axes[0, 1].set_xlabel("Time (sec)")
        axes[0, 1].set_ylabel("Frequency (Hz)")

        axes[1, 1].imshow(cleaned_spec, origin='lower', aspect='auto', cmap='magma', extent=save_extent)
        axes[1, 1].set_title(f"Final {SAVE_WINDOW_SEC}s clean Call (HDF5)")
        axes[1, 1].set_xlabel("Time (sec)")

        # Save the diagnostic plot
        plot_folder = os.path.join(specpath, 'diagnostic_plots')
        Path(plot_folder).mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{plot_folder}/call_{idx}_{onset:.2f}s.png")
        plt.close()



    # save to file when batch is full
    while len(syll_data['onsets']) >= sylls_per_file:
        save_filename = os.path.join(specpath, f"syllables_{write_file_num:04d}.hdf5")


        # write hd5f files with the different fields
        with h5py.File(save_filename, "w") as f:
             f.create_dataset("specs", data=np.stack(syll_data['specs'][:sylls_per_file])) #3D tesor [N, 128, 128]
             f.create_dataset("specs_raw", data=np.stack(syll_data['specs_raw'][:sylls_per_file])) #[N, 128, 128]
             f.create_dataset("onsets", data=np.array(syll_data['onsets'][:sylls_per_file])) #1D array [N]
             f.create_dataset("offsets", data=np.array(syll_data['offsets'][:sylls_per_file])) #1D array [N]
             f.create_dataset("audio_filenames", data=np.array([os.path.join(wavpath, fn) for fn in syll_data['audio_filenames'][:sylls_per_file]]).astype('S')) # array of strings [N]
             f.create_dataset("locations", data=np.array(syll_data['location'][:sylls_per_file]).astype('S')) # strings [N]


        # -- Clear written entries from buffer
        for key in syll_data:
            syll_data[key] = syll_data[key][sylls_per_file:]

        write_file_num += 1


# == After the main loop, save any remaining data in the buffer
if len(syll_data['onsets']) > 0:
    save_filename = os.path.join(specpath, f"syllables_{write_file_num:04d}_final.hdf5")
    with h5py.File(save_filename, "w") as f:
         for key in syll_data:
             if key == "audio_filenames" or key == "location":
                 f.create_dataset(key, data=np.array(syll_data[key]).astype('S'))
             elif key == "specs" or key == "specs_raw":
                 f.create_dataset(key, data=np.stack(syll_data[key]))
             else:
                 f.create_dataset(key, data=np.array(syll_data[key]))
    print(f"Saved final batch of {len(syll_data['onsets'])} syllables.")

print("Done!")