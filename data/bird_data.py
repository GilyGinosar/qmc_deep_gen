from torch.utils.data import Dataset
import os,glob
import h5py
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import torch
from gily_code.all.bin_specs_rule_based import classify_spectrogram_three_way

# --- fixed spectrogram grid (matches your preprocessing) ---
MIN_FREQ_HZ   = 500
MAX_FREQ_HZ   = 62500
NUM_FREQ_BINS = 128

def _onehot3_from_label(label: str) -> torch.Tensor:
    idx = {"low": 0, "alarm": 1, "high": 2}.get(label, 0)
    return torch.nn.functional.one_hot(torch.tensor(idx), num_classes=3).to(torch.float32)


def spec_to_tensor(x: np.ndarray) -> torch.Tensor:
    # x: (F, T) numpy array
    return torch.from_numpy(np.asarray(x)).to(torch.float32).unsqueeze(0)  # -> (1, F, T)


def load_segmented_sylls(bird_filepath,sylls,test_size=0.2,seed=92):

    spec_files = []
    syll_ids = []
    for syll in sylls:
        sub_path = os.path.join(bird_filepath,f'syll_specs_{syll}/*')
        
        syll_files = glob.glob(os.path.join(sub_path,'*.hdf5'))
        spec_files += syll_files
        syll_ids += [syll]*len(syll_files)
        
    train_files,test_files,train_ids,test_ids = train_test_split(spec_files,syll_ids,test_size=test_size,random_state=seed)
    
    return (train_files,test_files),(train_ids,test_ids)
        

class bird_data(Dataset):

    def __init__(self,filenames,syll_ids,specs_per_file=20,transform=spec_to_tensor,
                 conditional=False,conditional_factor='fm'):


        self.filenames=filenames
        self.syll_ids = syll_ids
        self.specs_per_file = specs_per_file
        self.transform = transform
        self.conditional=conditional
        self.conditional_factor = conditional_factor
        self.freq_axis = np.linspace(float(MIN_FREQ_HZ), float(MAX_FREQ_HZ),
                                     int(NUM_FREQ_BINS), dtype=np.float64)

        # Calculate length min and max for normalization
        if self.conditional and self.conditional_factor == 'length':
            self.min_dur = float('inf')
            self.max_dur = 0.0

            for fn in self.filenames:
                with h5py.File(fn, 'r', locking=False) as f:
                    on = f['onsets'][:]
                    off = f['offsets'][:]
                    durs = off - on  # shape (N,)

                    if durs.size == 0:
                        continue
                    self.min_dur = min(self.min_dur, float(durs.min()))
                    self.max_dur = max(self.max_dur, float(durs.max()))

            # fallback if something went wrong
            if not np.isfinite(self.min_dur) or self.max_dur <= self.min_dur:
                print("[bird_data:length] WARNING: invalid duration range; using [0,1] as dummy.")
                self.min_dur = 0.0
                self.max_dur = 1.0


    def __len__(self):
        return len(self.filenames) * self.specs_per_file


    def __getitem__(self,index):
        load_index = index // self.specs_per_file
        spec_index = index % self.specs_per_file
        load_fn = self.filenames[load_index]
        syll_id = self.syll_ids[load_index]

        with h5py.File(load_fn, 'r', locking=False) as f:
            spec = f['specs'][spec_index]

            if self.conditional:
                if self.conditional_factor == 'fm':
                    c = calc_fm(spec)
                elif self.conditional_factor == 'entropy':
                    c = calc_ent(spec)

                # elif self.conditional_factor == 'length':
                #     c = f['offsets'][spec_index] - f['onsets'][spec_index]

                elif self.conditional_factor == 'length':
                    # raw duration (same units as in the HDF5 file)
                    dur = f['offsets'][spec_index] - f['onsets'][spec_index]
                    dur = float(dur)

                    # --- DEBUG: print raw durations to understand units (samples? frames? sec?) ---
                    if hasattr(self, "_debug_len_count"):
                        pass
                    else:
                        self._debug_len_count = 0

                    if self._debug_len_count < 5:  # print only the first few
                        print(f"[DEBUG length] raw dur = {dur}")
                        print(f"  offset = {float(f['offsets'][spec_index])}")
                        print(f"  onset  = {float(f['onsets'][spec_index])}")
                        self._debug_len_count += 1
                    # ------------------------------------------------------------------------------

                    # normalize to [0,1] using dataset-wide min/max computed in __init__
                    rng = self.max_dur - self.min_dur
                    if rng <= 0:
                        dur_norm = 0  # degenerate fallback
                    else:
                        dur_norm = (dur - self.min_dur) / (rng + 1e-8)
                        dur_norm = float(np.clip(dur_norm, 0.0, 1.0))

                    # scalar tensor [1], compatible with cond_dim = 1
                    c = torch.tensor([dur_norm], dtype=torch.float32)





                elif self.conditional_factor == 'locations':
                    # analysis only (NOT for training)
                    c = f['locations'][spec_index].decode('ASCII')

                elif self.conditional_factor == 'file':
                    # analysis only (NOT for training)
                    c = f['audio_filenames'][spec_index].decode('ASCII')

                # elif self.conditional_factor == 'mean_freq':
                #     c = calc_mean_freq(spec)


                # ---- NEW one-hot binning (Option A: uses self.k22/self.k29 precomputed in __init__) ----
                # elif self.conditional_factor == 'mean_freq_bin1h':
                #     spec_np = np.array(spec, copy=False)
                #     c = calc_mean_freq_bin1h(spec_np, self.freq_axis)  # returns (3,) float32
                # elif self.conditional_factor == 'freq_bin1h':
                #     # generic: choose estimator name via attribute or hardcode here
                #     c = one_hot_bin1h(spec, self.freq_axis, estimator="median50", edges_hz=self.edges_hz)

                # elif self.conditional_factor == 'framecount':
                #     onehot, _ = framecount_mid_priority_bin(spec, self.freq_axis,
                #                                             edges_hz=self.edges_hz,
                #                                             use_ambiguous=False)
                #     c = onehot  # 1-hot np.float32
                # elif self.conditional_factor == 'framecount_argmax':
                #     c = framecount_argmax_bin(spec, self.freq_axis, edges_hz=self.edges_hz)

                elif self.conditional_factor == 'rule3_bands':
                    # spec: (F, T) numpy-like
                    S = np.asarray(spec, dtype=np.float64)
                    F = S.shape[0]

                    # Build a freq axis in **kHz** to match the classifier’s expectation.
                    # Use your fixed grid, but adapt if F != NUM_FREQ_BINS just in case.
                    f_hz = self.freq_axis
                    if f_hz.shape[0] != F:
                        f_hz = np.linspace(float(MIN_FREQ_HZ), float(MAX_FREQ_HZ), F, dtype=np.float64)
                    freqs_khz = f_hz / 1000.0

                    # Run your rule-based classifier (already imported at top)
                    label, _diag = classify_spectrogram_three_way(S, freqs_khz)

                    # One-hot torch tensor [3]: low=0, alarm=1, high=2
                    c = _onehot3_from_label(label)

                else:
                    raise NotImplementedError

        spec = self.transform(spec)

        if self.conditional:
            return (spec, c, syll_id)
        return (spec, syll_id)

    
def load_gerbils(gerbil_filepath,families=[2],test_size=0.2,seed=92,check=True):

    specs_per_file = 100
    try:
        len(families)
    except:
        families = [families]
    specs_in_file = []
    all_family_specs= {f:[] for f in families}
    all_family_ids = {f:[] for f in families}

    for ii,family in enumerate(families):
        print(f"loading family{family}")
        spec_dir = os.path.join(gerbil_filepath,'processed-data',f"family{family}")
        spec_fns = glob.glob(os.path.join(spec_dir,'*.hdf5'))
        all_family_specs[family] += spec_fns
        
        all_family_ids[family].append(family*np.ones((len(spec_fns),)))
        
        if check:
            for spec_fn in tqdm(spec_fns,total=len(spec_fns)):
                with h5py.File(spec_fn,'r') as f:
                    sif = len(f['specs'])
                    specs_in_file.append(sif)

    if check:
        num_specs = np.unique(specs_in_file)
        assert len(num_specs) == 1, print(f"Files have different numbers of specs in them! {num_specs}")
        if num_specs[0] != specs_per_file:
            print(f"expected {specs_per_file} specs per file, found {num_specs[0]}; updating")
            specs_per_file = num_specs[0]
    #all_family_ids = np.hstack(all_family_ids)
    #assert num_specs[0] == specs_per_file,print("num_specs,specs_per_file)
    if test_size > 0:
        train_fns,test_fns,train_ids,test_ids = [],[],[],[]
        for family in families:
            specs,ids = all_family_specs[family],np.hstack(all_family_ids[family])
            tr_fn,te_fn,tr_id,te_id = train_test_split(specs,ids,test_size=test_size,random_state=seed)
            train_fns.append(tr_fn)
            test_fns.append(te_fn)
            train_ids.append(tr_id)
            test_ids.append(te_id)
        #train_fns,test_fns,train_ids,test_ids = train_test_split(all_family_specs,all_family_ids,test_size=test_size,random_state=seed)
        train_fns = sum(train_fns,[])
        test_fns = sum(test_fns,[])
        train_ids = np.hstack(train_ids)
        test_ids = np.hstack(test_ids)
    else:
        train_fns,test_fns = all_family_specs, all_family_specs
        train_ids,test_ids = all_family_ids,all_family_ids
    #train_ids = np.zeros((len(train_fns,)))
    #test_ids = np.zeros((len(test_fns,)))

    return (train_fns,test_fns),(train_ids,test_ids),specs_per_file

#### song features from syllables

def calc_ent(spec):

    denom = np.sum(spec,axis=0,keepdims=True)#+1e-10)
    ps = spec/(denom + 1e-10)
    ent = -(np.log(ps + 1e-10) * ps).sum(axis=0)

    weights = (denom > 0).astype(np.float32)
    weights /= np.sum(weights)
    return (ent*weights.squeeze()).sum() #np.nanmean(ent)


def calc_fm(spec):
    """
    spec should be h x w bins
    
    """
    dt = np.diff(spec,axis=1)
    df = np.diff(spec,axis=0)
    dt2 = np.amax(dt**2,axis=0)
    df2 = np.amax(df**2,axis=0)
    fm =np.arctan(dt2,df2[:-1])
    weights = (np.sum(spec,axis=0) > 0).astype(np.float32)[:-1]
    weights /= np.sum(weights)
    return (fm * weights).sum()

# Gily's addition

_EPS = 1e-10

def _get_freq_axis(n_freq: int, freq_axis=None) -> np.ndarray:
    if freq_axis is None:
        return np.arange(n_freq, dtype=np.float64)
    f = np.asarray(freq_axis, dtype=np.float64)
    if f.shape[0] != n_freq:
        raise ValueError(f"freq_axis length {f.shape[0]} != n_freq {n_freq}")
    return f

def calc_mean_freq(spec: np.ndarray, freq_axis: np.ndarray | None = None) -> float:
    """
    Global energy-weighted mean frequency over the entire spectrogram.
    Matches the 'use raw spec as weights' spirit of calc_ent (no squaring),
    but collapses time first, then takes a single mean across freq.
    """
    S = np.asarray(spec, dtype=np.float64)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    F, T = S.shape
    f = _get_freq_axis(F, freq_axis)

    w_f = S.sum(axis=1)                 # energy per frequency bin, summed over time
    denom = w_f.sum()
    if denom <= _EPS:
        return 0.0
    return float((f * w_f).sum() / (denom + _EPS))





def calc_energy_weighted_median_hz(spec: np.ndarray,
                                   freq_axis: np.ndarray | None = None,
                                   time_reduce: str = "median") -> float:
    """
    50% spectral roll-off (energy-weighted median), no low-frequency trimming.

    Steps: time-aggregate to 1D spectrum (median/mean), normalize to weights,
    take the smallest bin where CDF >= 0.5. Returns Hz (or bin index if freq_axis is None).
    """
    S = np.asarray(spec, dtype=np.float64)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    F, T = S.shape
    # 1D spectrum across time
    if time_reduce == "mean":
        s = S.mean(axis=1)
    else:
        s = np.median(S, axis=1)  # robust default

    w = np.clip(s, 0, None)
    total = w.sum()
    if total <= _EPS:
        # fallback: mid-band (in Hz if freq_axis is provided, else mid index)
        if freq_axis is None:
            return float((F - 1) / 2.0)
        f = _get_freq_axis(F, freq_axis)
        return float(f[len(f)//2])

    cdf = np.cumsum(w / (total + _EPS))
    idx = int(np.searchsorted(cdf, 0.5, side="left"))
    idx = min(max(idx, 0), F - 1)

    if freq_axis is None:
        return float(idx)
    f = _get_freq_axis(F, freq_axis)
    return float(f[idx])



def _mean_row_index(spec_np: np.ndarray) -> float:
    """
    spec_np: (F, T) numpy array. Returns energy-weighted mean row index in [0, F-1].
    """
    import numpy as np
    F = spec_np.shape[0]
    w = spec_np.sum(axis=1)  # energy per frequency row
    s = w.sum()
    if s <= 0:
        return float(F - 1) / 2.0  # fallback: center
    idx = np.arange(F, dtype=np.float64)
    return float((w * idx).sum() / s)




# ---------- per-frame robust frequency ----------
def per_frame_rolloff_hz(spec_FT: np.ndarray,
                         freq_axis_hz: np.ndarray,
                         p: float = 0.50,
                         noise_percentile: float = 5.0,
                         min_frame_energy_ratio: float = 1e-4) -> np.ndarray:
    """
    For each time frame, compute p% roll-off frequency (Hz) after subtracting a
    per-frequency noise floor. Frames with too little energy are NaN.
    """
    S = np.asarray(spec_FT, dtype=np.float64)
    F, T = S.shape
    f = np.asarray(freq_axis_hz, dtype=np.float64); assert f.shape[0] == F

    # Per-frequency noise floor; subtract
    noise_f = np.percentile(S, noise_percentile, axis=1, keepdims=True)  # (F,1)
    X = np.maximum(S - noise_f, 0.0)

    # Frame energy + valid mask
    e_t = X.sum(axis=0) + 1e-12
    med_e = np.median(e_t) + 1e-12
    valid = e_t >= (min_frame_energy_ratio * med_e)

    # Cumulative sums per frame
    cum = np.cumsum(X, axis=0)        # (F,T)
    tot = cum[-1, :] + 1e-12          # (T,)
    thresh = p * tot

    out = np.full((T,), np.nan, dtype=np.float64)
    for t in range(T):
        if not valid[t]: continue
        k = int(np.searchsorted(cum[:, t], thresh[t], side="left"))
        k = min(max(k, 0), F - 1)
        out[t] = f[k]
    return out

# ---------- argmax ----------
# --- helpers (place near your other utils) ---
_EPS = 1e-10

def _one_hot(idx: int, n_classes: int = 3) -> np.ndarray:
    v = np.zeros((n_classes,), dtype=np.float32)
    if 0 <= idx < n_classes: v[idx] = 1.0
    return v

def _band_indices(freq_axis_hz: np.ndarray, edges_hz: tuple[float, float]):
    e1, e2 = float(edges_hz[0]), float(edges_hz[1])
    f = np.asarray(freq_axis_hz, dtype=np.float64)
    low_idx  = np.where(f < e1)[0]
    mid_idx  = np.where((f >= e1) & (f < e2))[0]
    high_idx = np.where(f >= e2)[0]
    return low_idx, mid_idx, high_idx

def framecount_argmax_bin(spec_FT: np.ndarray,
                          freq_axis_hz: np.ndarray,
                          edges_hz: tuple[float, float] = (22_000.0, 27_000.0),
                          alpha_gate: float = 0.05,
                          band_presence_frac: float = 0.25,
                          n_classes: int = 3) -> np.ndarray:
    """
    Multi-label presence per frame, then ARGMAX across bands at the clip level.
    Returns: (n_classes,) float32 one-hot
    """
    S = np.asarray(spec_FT, dtype=np.float64)
    S = np.maximum(S, 0.0)
    F, T = S.shape

    e_t = S.sum(axis=0)
    med = np.median(e_t) + 1e-12
    keep = e_t >= (alpha_gate * med)

    low_idx, mid_idx, high_idx = _band_indices(freq_axis_hz, edges_hz)

    cL = cM = cH = 0
    for t in range(T):
        if not keep[t]:
            continue
        col = S[:, t]
        E_low  = float(col[low_idx].sum())  if low_idx.size  else 0.0
        E_mid  = float(col[mid_idx].sum())  if mid_idx.size  else 0.0
        E_high = float(col[high_idx].sum()) if high_idx.size else 0.0
        tot = E_low + E_mid + E_high
        if tot <= 0.0:
            continue
        if (E_low  / tot) >= band_presence_frac:  cL += 1
        if (E_mid  / tot) >= band_presence_frac:  cM += 1
        if (E_high / tot) >= band_presence_frac:  cH += 1

    counts = np.array([cL, cM, cH], dtype=int)
    idx = int(np.argmax(counts))  # pure argmax
    return _one_hot(idx, n_classes).astype(np.float32)




# ---------- bin utils ----------

def _bin_index_from_edges(val_hz: float,
                          edges_hz: tuple[float, float]) -> int:
    """
    With two edges (e1,e2): 0 if f<e1, 1 if e1<=f<e2, 2 if f>=e2
    """
    e1, e2 = float(edges_hz[0]), float(edges_hz[1])
    if val_hz < e1: return 0
    if val_hz < e2: return 1
    return 2

# older version
# def one_hot_bin1h(spec: np.ndarray,
#                   freq_axis: np.ndarray,
#                   estimator: str = "median50",
#                   edges_hz: tuple[float, float] = (22_000.0, 25_000.0),
#                   n_classes: int = 3) -> np.ndarray:
#     """
#     General 1-hot binning using any scalar frequency estimator.
#
#     estimator:
#       - "median50"      -> calc_energy_weighted_median_hz
#       - "mean"          -> calc_mean_freq
#       - "median_global" -> calc_median_freq          (your existing global-median implementation)
#       - "avg_over_time" -> calc_avg_freq_over_time   (your equal-time mean)
#
#     Returns: (n_classes,) float32 one-hot vector.
#     """
#     if estimator == "median50":
#         f_hz = calc_energy_weighted_median_hz(spec, freq_axis=freq_axis)
#     elif estimator == "mean":
#         f_hz = calc_mean_freq(spec, freq_axis=freq_axis)
#     else:
#         raise ValueError(f"Unknown estimator '{estimator}'")
#
#     idx = _bin_index_from_edges(float(f_hz), edges_hz)
#     return _one_hot(idx, n_classes).astype(np.float32)

# counting frames
def framecount_mid_priority_bin(spec_FT: np.ndarray,
                                freq_axis_hz: np.ndarray,
                                edges_hz: tuple[float, float] = (22_000.0, 27_000.0),
                                p_rolloff: float = 0.50,
                                noise_percentile: float = 5.0,
                                min_frame_energy_ratio: float = 1e-4,
                                min_valid_frames: int = 3,
                                # MID priority threshold:
                                mid_priority_frac: float = 0.25,    # “good amount” of mid frames
                                # optional ambiguous bucket:
                                use_ambiguous: bool = False,
                                ambiguous_margin: float = 0.10,
                                n_classes_if_ambiguous: int = 4):
    """
    Bin decision with MID priority:
      1) If fraction of valid frames in MID >= mid_priority_frac -> MID (regardless of HIGH).
      2) Else if frames ONLY in HIGH -> HIGH.
      3) Else if LOW has the most frames -> LOW.
      4) Else if use_ambiguous and all three are comparable -> AMBIGUOUS.
      5) Else -> argmax among (LOW, MID, HIGH).

    Returns: (onehot, info_dict)
    """
    # 1) per-frame robust frequency
    fpf = per_frame_rolloff_hz(
        spec_FT, freq_axis_hz,
        p=p_rolloff,
        noise_percentile=noise_percentile,
        min_frame_energy_ratio=min_frame_energy_ratio,
    )
    good = np.isfinite(fpf)
    n_valid = int(good.sum())

    if n_valid < min_valid_frames:
        if use_ambiguous:
            return _one_hot(3, n_classes_if_ambiguous), {"valid_frames": n_valid, "counts": (0,0,0), "rule": "too_few_valid"}
        return _one_hot(0, 3), {"valid_frames": n_valid, "counts": (0,0,0), "rule": "too_few_valid"}  # default to low

    # 2) counts per bin
    bins = np.array([_bin_index_from_edges(x, edges_hz) for x in fpf[good]], dtype=int)
    counts = np.bincount(bins, minlength=3)
    total = int(counts.sum())
    fracs = counts / (total if total > 0 else 1)

    low, mid, high = int(counts[0]), int(counts[1]), int(counts[2])
    fL, fM, fH = float(fracs[0]), float(fracs[1]), float(fracs[2])

    # ---- decision rules (ordered) ----

    # (1) MID priority
    if fM >= mid_priority_frac:
        if use_ambiguous:
            return _one_hot(1, n_classes_if_ambiguous), {"valid_frames": n_valid, "counts": (low, mid, high), "rule": "mid_priority"}
        return _one_hot(1, 3), {"valid_frames": n_valid, "counts": (low, mid, high), "rule": "mid_priority"}

    # (2) HIGH-only
    if low == 0 and mid == 0 and high > 0:
        if use_ambiguous:
            return _one_hot(2, n_classes_if_ambiguous), {"valid_frames": n_valid, "counts": (low, mid, high), "rule": "only_high"}
        return _one_hot(2, 3), {"valid_frames": n_valid, "counts": (low, mid, high), "rule": "only_high"}

    # (3) LOW majority
    if (low > mid) and (low >= high):
        if use_ambiguous:
            return _one_hot(0, n_classes_if_ambiguous), {"valid_frames": n_valid, "counts": (low, mid, high), "rule": "low_majority"}
        return _one_hot(0, 3), {"valid_frames": n_valid, "counts": (low, mid, high), "rule": "low_majority"}

    # (4) optional AMBIGUOUS: all comparable (likely noise)
    if use_ambiguous:
        spread = max(fL, fM, fH) - min(fL, fM, fH)
        if spread <= ambiguous_margin:
            return _one_hot(3, n_classes_if_ambiguous), {"valid_frames": n_valid, "counts": (low, mid, high), "rule": "ambiguous_all_comparable"}

    # (5) fallback: argmax (ties → np.argmax order)
    winner = int(np.argmax(counts))
    if use_ambiguous:
        return _one_hot(winner, n_classes_if_ambiguous), {"valid_frames": n_valid, "counts": (low, mid, high), "rule": "argmax"}
    return _one_hot(winner, 3), {"valid_frames": n_valid, "counts": (low, mid, high), "rule": "argmax"}

