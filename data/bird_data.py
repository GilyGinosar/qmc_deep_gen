from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os,glob
import h5py
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm 

# --- fixed spectrogram grid (matches your preprocessing) ---
MIN_FREQ_HZ   = 500
MAX_FREQ_HZ   = 62500
NUM_FREQ_BINS = 128

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

    def __init__(self,filenames,syll_ids,specs_per_file=20,transform=transforms.ToTensor(),
                 conditional=False,conditional_factor='fm'):


        self.filenames=filenames
        self.syll_ids = syll_ids
        self.specs_per_file = specs_per_file
        self.transform = transform
        self.conditional=conditional
        self.conditional_factor = conditional_factor

        # NEW: precompute linear frequency axis + cut indices for 22k/29k
        self.freq_axis = np.linspace(float(MIN_FREQ_HZ), float(MAX_FREQ_HZ),
                                     int(NUM_FREQ_BINS), dtype=np.float64)
        self.k22 = int(np.argmin(np.abs(self.freq_axis - 22000.0)))
        self.k29 = int(np.argmin(np.abs(self.freq_axis - 29000.0)))


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
                elif self.conditional_factor == 'length':
                    c = f['offsets'][spec_index] - f['onsets'][spec_index]
                elif self.conditional_factor == 'locations':
                    # analysis only (NOT for training)
                    c = f['locations'][spec_index].decode('ASCII')
                elif self.conditional_factor == 'file':
                    # analysis only (NOT for training)
                    c = f['audio_filenames'][spec_index].decode('ASCII')

                elif self.conditional_factor == 'mean_freq':
                    c = calc_mean_freq(spec)
                elif self.conditional_factor == 'median_freq':
                    c = calc_median_freq(spec)
                elif self.conditional_factor == 'avg_freq_over_time':
                    c = calc_avg_freq_over_time(spec)

                # ---- NEW one-hot binning (Option A: uses self.k22/self.k29 precomputed in __init__) ----
                elif self.conditional_factor == 'mean_freq_bin1h':
                    spec_np = np.array(spec, copy=False)
                    c = calc_mean_freq_bin1h(spec_np, self.freq_axis)  # returns (3,) float32

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

def calc_median_freq(spec: np.ndarray, freq_axis: np.ndarray | None = None) -> float:
    """
    Global energy-weighted *median* frequency:
    the frequency where cumulative SUM over time+freq reaches 50%.
    This mirrors calc_mean_freq’s 'global' viewpoint but uses the median (robust).
    """
    S = np.asarray(spec, dtype=np.float64)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    F, T = S.shape
    f = _get_freq_axis(F, freq_axis)

    w_f = S.sum(axis=1)                 # (F,)
    total = w_f.sum()
    if total <= _EPS:
        return 0.0

    cdf = np.cumsum(w_f) / (total + _EPS)
    idx = int(np.searchsorted(cdf, 0.5, side="left"))
    idx = min(max(idx, 0), F - 1)
    return float(f[idx])

def calc_avg_freq_over_time(spec: np.ndarray, freq_axis: np.ndarray | None = None) -> float:
    """
    Equal-time average of per-frame mean frequency.
    For each time frame, normalize over frequency (like calc_ent),
    compute the mean frequency of that frame, then average across all
    non-empty frames with *equal* weight per frame.
    """
    S = np.asarray(spec, dtype=np.float64)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    F, T = S.shape
    f = _get_freq_axis(F, freq_axis)

    # per-frame totals and non-empty mask (same idea as calc_ent)
    denom_t = S.sum(axis=0)                     # (T,)
    non_empty = denom_t > _EPS

    if not np.any(non_empty):
        return 0.0

    # per-frame probabilities over frequency, then per-frame mean frequency
    # (compute only for non-empty frames to avoid divide-by-zero)
    frame_means = []
    for t in np.where(non_empty)[0]:
        p_ft = S[:, t] / (denom_t[t] + _EPS)    # (F,)
        frame_means.append(float((f * p_ft).sum()))

    # equal-time average (uniform over non-empty frames)
    return float(np.mean(frame_means))


def _freq_bin_idx_22_29(f_hz: float) -> int:
    """Return 0 if <22k, 1 if [22k,29k), 2 if >=29k."""
    if f_hz < 22_000.0: return 0
    if f_hz < 29_000.0: return 1
    return 2

def _one_hot(idx: int, n: int = 3) -> np.ndarray:
    v = np.zeros((n,), dtype=np.float32)
    if 0 <= idx < n:
        v[idx] = 1.0
    return v

def calc_mean_freq_bin1h(spec: np.ndarray, freq_axis: np.ndarray) -> np.ndarray:
    f = calc_mean_freq(spec, freq_axis=freq_axis)       # returns Hz
    return _one_hot(_freq_bin_idx_22_29(float(f)), 3)

def calc_median_freq_bin1h(spec: np.ndarray, freq_axis: np.ndarray) -> np.ndarray:
    f = calc_median_freq(spec, freq_axis=freq_axis)     # returns Hz
    return _one_hot(_freq_bin_idx_22_29(float(f)), 3)

def calc_avg_freq_over_time_bin1h(spec: np.ndarray, freq_axis: np.ndarray) -> np.ndarray:
    f = calc_avg_freq_over_time(spec, freq_axis=freq_axis)  # returns Hz
    return _one_hot(_freq_bin_idx_22_29(float(f)), 3)

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