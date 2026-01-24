import pandas as pd
import numpy as np
from scipy.io import wavfile
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
exp = 235

data_path = fr"\\sanesstorage.cns.nyu.edu\archive\ginosar\Processed_data\Audio\{exp}" #'/mnt/home/mmartinez/ceph/data/gerbil/gily'
print(data_path)
onoffpath = os.path.join(data_path,'vox_for_Miles.csv')
wavpath = os.path.join(data_path,'Averaged_wavs_w_annotations') #'audio')
specpath = os.path.join(data_path,'processed-data/family1_21_1')

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
SAVE_WINDOW_SEC = 0.5   # Final window saved to HDF5

write_file_num = 0
syll_data = {
    'specs':[], # masked (cleaned)
    'specs_raw':[], # unmasked
    'onsets':[],
    'offsets':[],
    'audio_filenames':[],
    'location':[]
}
sylls_per_file = params['sylls_per_file']
threshold_db = -60

# == Main loop
#for onset, offset, fn, loc in tqdm(zip(onsets, offsets, filenames, locs), total=len(filenames)):
for idx, (onset, offset, fn, loc) in enumerate(tqdm(zip(onsets, offsets, filenames, locs), total=len(filenames))):
    # -- 1. Calculate the context window for cleaning
    midpoint = (onset + offset) / 2
    m_start = midpoint - (MASK_CONTEXT_SEC / 2)
    m_end = midpoint + (MASK_CONTEXT_SEC / 2)

    # -- 2. Extract context spectrogram
    spec_list, valid = get_syll_specs([m_start], [m_end], fn, params)


    if valid:
        print(f"Processing call {idx}")
        # get_syll_specs returns a list; take the first spectrogram
        spec_2s = spec_list[0]

        # -- 3. Generate the Mask using the 2s context
        # Normalize 0-1 for the engine
        spec_norm = (spec_2s - spec_2s.min()) / (spec_2s.max() - spec_2s.min())
        mask_2s = apply_zonal_cleaning(spec_norm, params['fs'])

        # -- 4. Apply Mask to the raw 2s data
        cleaned_2s = spec_2s * mask_2s

        # -- 5. Calculate crop indices
        # Find how many columns represent 0.5s out of 2.0s
        total_cols = spec_2s.shape[1]
        center_col = total_cols // 2
        # half_win is 0.25s worth of pixels
        half_win = int((SAVE_WINDOW_SEC / MASK_CONTEXT_SEC) * (total_cols / 2))

        # Slicing the arrays
        final_cleaned = cleaned_2s[:, center_col - half_win: center_col + half_win]
        final_raw = spec_2s[:, center_col - half_win: center_col + half_win]


        # -- 6. Append to buffer
        syll_data['specs'] += [final_cleaned]
        syll_data['specs_raw'] += [final_raw] # Changed to match your init
        syll_data['onsets'] += [onset]
        syll_data['offsets'] += [offset]
        syll_data['audio_filenames'] += [os.path.split(fn)[-1]]
        syll_data['location'] += [loc]

        # -- 7. NEW: Comparison Plotting
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Row 1: The 2.0s Context
        axes[0, 0].imshow(spec_2s, origin='lower', aspect='auto', cmap='magma')
        axes[0, 0].set_title(f"Raw 2.0s Context (Call {idx})")

        axes[0, 1].imshow(cleaned_2s, origin='lower', aspect='auto', cmap='magma')
        axes[0, 1].set_title("Masked 2.0s Context")

        # Row 2: The 0.5s HDF5 Final Slices
        axes[1, 0].imshow(final_raw, origin='lower', aspect='auto', cmap='magma')
        axes[1, 0].set_title("Final 0.5s Raw (HDF5)")

        axes[1, 1].imshow(final_cleaned, origin='lower', aspect='auto', cmap='magma')
        axes[1, 1].set_title("Final 0.5s Masked (HDF5)")

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