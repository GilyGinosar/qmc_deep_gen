
from data.bird_data import load_gerbils,bird_data
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import os, glob, h5py
from sklearn.model_selection import train_test_split
from tqdm import tqdm



# Gily - Becasue I'm using Windows, can't use Miles's lambdas
def spec_to_tensor(x: np.ndarray) -> torch.Tensor:
    # x shape: H x W numpy array
    return torch.from_numpy(x).to(torch.float32).unsqueeze(0)

def load_gerbils_multi(gerbil_filepath, specs_per_file, families=[2],
                 test_size=0.2, seed=92, check=True):
    """
            1: [r"D:\Data\235\alarms", r"D:\Data\237\alarms"],
            2: [r"D:\Data\112\alarms", r"D:\Data\115\alarms", r"D:\Data\116\alarms"]
    """
    try:
        len(families)
    except Exception:
        families = [families]

    # Normalize gerbil_filepath into a "get_roots(family)" accessor
    if isinstance(gerbil_filepath, dict):
        def get_roots(fam):  # per-family explicit roots
            return gerbil_filepath.get(fam, [])
    elif isinstance(gerbil_filepath, (list, tuple)):
        def get_roots(fam):  # same set of roots for all families
            return list(gerbil_filepath)
    else:
        def get_roots(fam):  # single root (original)
            return [gerbil_filepath]

    specs_in_file = []
    all_family_specs = []
    all_family_ids = []

    for ii, family in enumerate(families):
        print(f"loading family{family}")
        roots = get_roots(family)

        fam_spec_fns = []
        for root in roots:
            spec_dir = os.path.join(root, 'processed-data', f"family{family}")
            spec_fns = glob.glob(os.path.join(spec_dir, '*.hdf5'))
            fam_spec_fns.extend(spec_fns)

            if check:
                for spec_fn in tqdm(spec_fns, total=len(spec_fns), desc=f"checking {spec_dir}"):
                    with h5py.File(spec_fn, 'r') as f:
                        sif = len(f['specs'])
                        specs_in_file.append(sif)

        # accumulate
        all_family_specs += fam_spec_fns
        all_family_ids.append(ii * np.ones((len(fam_spec_fns),)))  # keep original labeling (0..K-1)

    if check and specs_in_file:
        num_specs = np.unique(specs_in_file)
        assert len(num_specs) == 1, print(f"Files have different numbers of specs in them! {num_specs}")
        if num_specs[0] != specs_per_file:
            print(f"expected {specs_per_file} specs per file, found {num_specs[0]}; updating")
            specs_per_file = int(num_specs[0])

    all_family_ids = np.hstack(all_family_ids) if len(all_family_ids) else np.array([])

    if test_size > 0 and len(all_family_specs) > 0:
        train_fns, test_fns, train_ids, test_ids = train_test_split(
            all_family_specs, all_family_ids, test_size=test_size, random_state=seed
        )
    else:
        train_fns, test_fns = all_family_specs, all_family_specs
        train_ids, test_ids = all_family_ids, all_family_ids

    return (train_fns, test_fns), (train_ids, test_ids), specs_per_file




if __name__ == "__main__":

    n_workers = max(os.cpu_count()-1,1) #len(os.sched_getaffinity(0))
    n_workers = 0
    print(n_workers)

    # Combine families across experiments
    roots_per_family = {
        1: [r"D:\Data\235",r"D:\Data\237"],
        #2: [r"D:\Data\113", r"D:\Data\114", r"D:\Data\115", r"D:\Data\116"], #r"D:\Data\112",
    }
    out_dir = fr"D:\data\model_checkpoints"
    os.makedirs(out_dir, exist_ok=True)



    # ----- load files with spectrograms ------
    ### specs_per_file is how many spectrograms are in each .hdf5 file, all files have 100 vocalization each, families is the family number we're trying to load,
    ### test_size - portion of the data that will remain unseen in training, seed is used to maintain reproducibility, check determines whether we check to see if
    (train_fns, test_fns), (train_ids, test_ids), specs_per_file = load_gerbils_multi(gerbil_filepath=roots_per_family, specs_per_file=100, families=[1], test_size=0.2, seed=92, check=True)


    # ----- load datasets (train / test) : single samples -----

    # --- BIN THE DATA ONCE using the rule-based conditional dataset ---
    # We only use this to get the bin index per item.
    train_ds_cond = bird_data(
        train_fns, train_ids, specs_per_file=specs_per_file,
        transform=spec_to_tensor, conditional=True, conditional_factor="rule3_bands"
    )
    test_ds_cond = bird_data(
        test_fns, test_ids, specs_per_file=specs_per_file,
        transform=spec_to_tensor, conditional=True, conditional_factor="rule3_bands"
    )

    # Plain (unconditional) datasets used for training each model
    train_ds_plain = bird_data(
        train_fns, train_ids, specs_per_file=specs_per_file,
        transform=spec_to_tensor, conditional=False
    )
    test_ds_plain = bird_data(
        test_fns, test_ids, specs_per_file=specs_per_file,
        transform=spec_to_tensor, conditional=False
    )

    # Collect indices per bin (0=low, 1=alarm, 2=high)
    bin_to_idx_tr = {0: [], 1: [], 2: []}
    for i in range(len(train_ds_cond)):
        _, c, _ = train_ds_cond[i]
        bin_to_idx_tr[int(c.argmax().item())].append(i)

    bin_to_idx_te = {0: [], 1: [], 2: []}
    for i in range(len(test_ds_cond)):
        _, c, _ = test_ds_cond[i]
        bin_to_idx_te[int(c.argmax().item())].append(i)

    from torch.utils.data import Subset

    # Build per-bin loaders on the PLAIN datasets (unconditional)
    BATCH_UNCOND = 64
    use_cuda = torch.cuda.is_available()
    pin = bool(use_cuda)


    from torch.utils.data import Subset

    loaders_per_bin = {}
    for b in (0, 1, 2):
        tr_subset = Subset(train_ds_plain, bin_to_idx_tr[b])
        te_subset = Subset(test_ds_plain, bin_to_idx_te[b])
        ntr, nte = len(tr_subset), len(te_subset)

        if ntr == 0:
            print(f"BIN {b}: empty (train=0, test={nte}) — skipping.")
            loaders_per_bin[b] = {"train": None, "test": None, "n_train": 0, "n_test": nte}
            continue

        bs_tr = min(BATCH_UNCOND, ntr)
        bs_te = min(BATCH_UNCOND, max(1, nte))

        loaders_per_bin[b] = dict(
            train=DataLoader(tr_subset, batch_size=bs_tr, shuffle=True,
                             num_workers=n_workers, pin_memory=pin),
            test=DataLoader(te_subset, batch_size=bs_te, shuffle=False,
                            num_workers=n_workers, pin_memory=pin),
            n_train=ntr,
            n_test=nte,
        )

    print({b: (v["n_train"], v["n_test"]) for b, v in loaders_per_bin.items()})

    # # ----- DBG
    # from collections import Counter
    # cnt = Counter()
    # for _, c, _ in DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=0):
    #     idx = c.argmax(dim=1).tolist()   # c is one-hot
    #     cnt.update(idx)
    # print("[train class counts]", dict(cnt))

    from models.sampling import gen_fib_basis,gen_korobov_basis
    from models.utils import get_decoder_arch
    from models.qmc_base import QMCLVM

    latent_dim=2 # sets our latent dimension
    ### If we use two dimensions, we should use gen_fib_basis for our grid over the latent space
    ### If more than two dimensions, we should use gen_korobov_basis. This requires additional arguments,
    ### if you want to use this see help(gen_korobov_basis) for good argument values

    latent_grid = gen_fib_basis(m=15) # m determines both the size of our grid and spacing of points
    ## if you want to plot this, you will need to plot (latent_grid % 1) instead of latent_grid


    dataset = 'gerbil_ava' # used for getting a pre-selected decoder architecture
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # use gpu if possible
    # make sure I'm on gpu
    import torch

    print(f"[DEVICE] Using: {device}")
    if device.type == "cuda":
        print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[DEVICE] CUDA capability: {torch.cuda.get_device_capability(0)}")
        print(f"[DEVICE] cuDNN enabled: {torch.backends.cudnn.enabled}")
    from train.losses import binary_evidence, binary_lp
    from train.train import train_loop

    qmc_loss_func = binary_evidence
    qmc_lp = binary_lp
    nEpochs = 1

    # Train three separate UNCONDITIONAL models (one per bin)
    for b in (0, 1, 2):
        print(f"\n=== Training UNCONDITIONAL model on BIN {b} ===")

        # Unconditional decoder; if you don't have 'qmc', you can still use 'conditional_qmc'
        # with cond_dim=0, but most repos have a plain 'qmc' arch for unconditional training.
        decoder = get_decoder_arch(dataset_name=dataset, latent_dim=latent_dim, arch="qmc")
        model = QMCLVM(latent_dim=latent_dim, device=device, decoder=decoder)

        # (optional) quick check of first linear
        first_linear = None
        for m in model.decoder.modules():
            if isinstance(m, nn.Linear):
                first_linear = m
                break
        if first_linear is not None:
            print("[CHECK] decoder[0]:", first_linear)

        bin_out_dir = os.path.join(out_dir, f"bin{b}")
        os.makedirs(bin_out_dir, exist_ok=True)

        model, opt, losses = train_loop(
            model,
            loaders_per_bin[b]["train"],  # <-- per-bin UNCONDITIONAL loader
            latent_grid.to(device),
            qmc_loss_func,
            nEpochs=nEpochs,
            verbose=True,
            conditional=False,  # <-- key: UNCONDITIONAL
            out_dir=bin_out_dir
        )

        # (optional) evaluate on that bin’s test set
        from train.train import test_epoch

        _ = test_epoch(model, loaders_per_bin[b]["test"], latent_grid.to(device),
                       qmc_loss_func, conditional=False)

