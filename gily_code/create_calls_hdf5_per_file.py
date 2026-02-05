import pandas as pd
import numpy as np
from scipy.io import wavfile
from skimage.transform import resize
import os
from tqdm import tqdm
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

from gily_code.ava_utils import get_spec
from ava.preprocessing.preprocess import get_syll_specs
from gily_code.clean_audio import apply_zonal_cleaning

### ---------------- Paths and parameters ----------------
exp = 275

data_path = fr"\\sanesstorage.cns.nyu.edu\archive\ginosar\Processed_data\Audio\{exp}"
print(data_path)
onoffpath = os.path.join(data_path, "vox_for_qlvm.csv")
wavpath = os.path.join(data_path, "Averaged_wavs_w_annotations")
specpath = os.path.join(data_path, "processed-data")

Path(specpath).mkdir(parents=True, exist_ok=True)

params = {
    "get_spec": get_spec,  # spectrogram maker
    "num_freq_bins": 128,
    "num_time_bins": 128,
    "min_freq": 500,  # minimum frequency
    "max_dur": 0.5,  # maximum syllable duration
    "max_freq": 62500,  # maximum frequency
    "nperseg": 512,  # FFT
    "noverlap": 256,  # FFT
    "spec_min_val": -8,  # minimum log-spectrogram value
    "spec_max_val": -5,  # maximum log-spectrogram value
    "fs": 125000,  # audio sample rate
    "mel": False,  # frequency spacing, mel or linear
    "time_stretch": True,  # stretch short syllables?
    "within_syll_normalize": False,  # normalize spectrogram values per spec
    "max_num_syllables": None,  # maximum number of syllables per directory
    "sylls_per_file": 100,  # unused here; kept for parity
    "real_preprocess_params": (
        "min_freq",
        "max_freq",
        "spec_min_val",
        "spec_max_val",
        "max_dur",
    ),
    "int_preprocess_params": ("nperseg", "noverlap"),
    "binary_preprocess_params": ("time_stretch", "mel", "within_syll_normalize"),
}

def should_skip_empty_cleaned(cleaned_spec):
    # Isolated hook for future handling of empty cleaned calls.
    return np.all(cleaned_spec == 0)

### ---------------- Prepare data lists ----------------
voc_data = pd.read_csv(onoffpath)
voc_data = voc_data.copy()
voc_data["file_num"] = voc_data["file_num"].astype(int)

### ---------------- Main processing ----------------
MASK_CONTEXT_SEC = 2.0  # window used for mask calculation
SAVE_WINDOW_SEC = 0.3   # final window saved to HDF5
BUFFER_SEC = 0.02       # 20ms padding around the actual call
MIN_CTX_SAMPLES = 1024  # skip tiny contexts
CLEAN_EVENT_TYPES = {"warble","ufm", "hat", "tilda"}

grouped = voc_data.groupby("file_num")

for file_num, df_file in tqdm(grouped, total=len(grouped)):
    stats = {
        "rows": 0,
        "missing_wav": 0,
        "edge_skip": 0,
        "short_ctx": 0,
        "flat_ctx": 0,
        "clean_fail": 0,
        "empty_cleaned": 0,
        "invalid_spec": 0,
        "saved": 0,
    }
    syll_data = {
        "specs": [],
        "specs_raw": [],
        "onsets": [],
        "offsets": [],
        "audio_filenames": [],
        "locations": [],
        "call_type": [],
        "cleaned": [],
    }

    # Process each channel within the same file number
    for channel, df_chan in df_file.groupby("assigned_channel"):
        stats["rows"] += len(df_chan)
        wav_path = os.path.join(
            wavpath, f"channel_{channel}_file_{str(file_num).zfill(3)}.wav"
        )
        if not os.path.exists(wav_path):
            print(f"Missing wav: {wav_path}")
            stats["missing_wav"] += len(df_chan)
            continue

        fs, audio = wavfile.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        total_dur = len(audio) / fs

        for onset, offset, loc, event_type in zip(
            df_chan.start_time_file_sec,
            df_chan.stop_time_file_sec,
            df_chan.assigned_location,
            df_chan["event_type"],
        ):
            event_key = str(event_type).strip().lower()
            if event_key == "noise":
                continue
            # -- 1. Gatekeeper: Skip Edge Calls
            midpoint = (onset + offset) / 2
            ml_start = midpoint - (SAVE_WINDOW_SEC / 2)
            ml_end = midpoint + (SAVE_WINDOW_SEC / 2)
            if ml_start < 0 or ml_end > total_dur:
                stats["edge_skip"] += 1
                continue

            # -- 2. Cleaning
            c_start = max(0, midpoint - (MASK_CONTEXT_SEC / 2))
            c_end = min(total_dur, midpoint + (MASK_CONTEXT_SEC / 2))
            audio_ctx = audio[int(c_start * fs) : int(c_end * fs)]
            if len(audio_ctx) < MIN_CTX_SAMPLES:
                stats["short_ctx"] += 1
                continue

            p_ctx, _, bins_ctx, _ = plt.specgram(audio_ctx, NFFT=512, Fs=fs, noverlap=256)
            plt.close()

            spec_db = 10 * np.log10(p_ctx + 1e-10)
            denom = spec_db.max() - spec_db.min()
            if not np.isfinite(denom) or denom <= 0:
                stats["flat_ctx"] += 1
                continue
            ctx_norm = (spec_db - spec_db.min()) / denom
            did_clean = event_key in CLEAN_EVENT_TYPES
            if did_clean:
                try:
                    mask_ctx = apply_zonal_cleaning(ctx_norm, fs)
                except Exception as e:
                    print(f"clean_fail {wav_path} onset={onset:.3f}: {e}")
                    stats["clean_fail"] += 1
                    continue
            else:
                mask_ctx = np.ones_like(ctx_norm)

            # -- 3. Create the "Call-Only" Mask
            local_onset = (onset - c_start) - BUFFER_SEC
            local_offset = (offset - c_start) + BUFFER_SEC

            call_only_mask = np.zeros_like(mask_ctx)
            call_cols = np.where((bins_ctx >= local_onset) & (bins_ctx <= local_offset))[0]
            call_only_mask[:, call_cols] = mask_ctx[:, call_cols]

            # -- 4. Extract ML Snippet & Apply Mask
            spec_list, valid = get_syll_specs([ml_start], [ml_end], wav_path, params)
            if not valid:
                stats["invalid_spec"] += 1
                continue

            raw_spec = spec_list[0]

            # Crop the call-only mask to the 0.3s window
            actual_dur = c_end - c_start
            center_col = int(((midpoint - c_start) / actual_dur) * mask_ctx.shape[1])
            hw = int((SAVE_WINDOW_SEC / actual_dur) * (mask_ctx.shape[1] / 2))
            cropped_mask = call_only_mask[:, center_col - hw : center_col + hw]

            # Resize and multiply
            final_mask = resize(
                cropped_mask,
                (params["num_freq_bins"], params["num_time_bins"]),
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            )
            cleaned_spec = raw_spec * final_mask

            # -- 4b. Skip empty cleaned calls (easy to tweak later)
            if did_clean and should_skip_empty_cleaned(cleaned_spec):
                stats["empty_cleaned"] += 1
                continue

            syll_data["specs"].append(cleaned_spec)
            syll_data["specs_raw"].append(raw_spec)
            syll_data["onsets"].append(onset)
            syll_data["offsets"].append(offset)
            syll_data["audio_filenames"].append(wav_path)
            syll_data["locations"].append(loc)
            syll_data["call_type"].append(event_key)
            syll_data["cleaned"].append(did_clean)
            stats["saved"] += 1

    num_specs = len(syll_data["specs"])
    if num_specs == 0:
        print(
            f"file_num {file_num:03d} -> rows={stats['rows']}, missing_wav={stats['missing_wav']}, "
            f"edge_skip={stats['edge_skip']}, short_ctx={stats['short_ctx']}, flat_ctx={stats['flat_ctx']}, "
            f"clean_fail={stats['clean_fail']}, empty_cleaned={stats['empty_cleaned']}, "
            f"invalid_spec={stats['invalid_spec']}, saved={stats['saved']}"
        )
        continue

    save_filename = os.path.join(specpath, f"file_{str(file_num).zfill(3)}.hdf5")

    with h5py.File(save_filename, "w") as f:
        f.create_dataset("specs", data=np.stack(syll_data["specs"]))
        f.create_dataset("specs_raw", data=np.stack(syll_data["specs_raw"]))
        f.create_dataset("onsets", data=np.array(syll_data["onsets"]))
        f.create_dataset("offsets", data=np.array(syll_data["offsets"]))
        f.create_dataset("audio_filenames", data=np.array(syll_data["audio_filenames"]).astype("S"))
        f.create_dataset("locations", data=np.array(syll_data["locations"]).astype("S"))
        f.create_dataset("call_type", data=np.array(syll_data["call_type"]).astype("S"))
        f.create_dataset("cleaned", data=np.array(syll_data["cleaned"], dtype=np.bool_))
        f.create_dataset("num_specs", data=np.array(num_specs, dtype=np.int32))

    print(
        f"file_num {file_num:03d} -> rows={stats['rows']}, missing_wav={stats['missing_wav']}, "
        f"edge_skip={stats['edge_skip']}, short_ctx={stats['short_ctx']}, flat_ctx={stats['flat_ctx']}, "
        f"clean_fail={stats['clean_fail']}, empty_cleaned={stats['empty_cleaned']}, "
        f"invalid_spec={stats['invalid_spec']}, saved={stats['saved']}"
    )

print("Done!")
