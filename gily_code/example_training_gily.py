
from data.bird_data import load_gerbils,bird_data
from torch.utils.data import DataLoader
import numpy as np
import os
import torch

# Gily - Becasue I'm using Windows, can't use Miles's lambdas
def spec_to_tensor(x: np.ndarray) -> torch.Tensor:
    # x shape: H x W numpy array
    return torch.from_numpy(x).to(torch.float32).unsqueeze(0)

# Gily - Because I'm combining same family over multiple experiments
import os, glob, h5py
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_gerbils_multi(gerbil_filepath, specs_per_file, families=[2],
                 test_size=0.2, seed=92, check=True):

    """
    gerbil_filepath can be:
      - str: a single root used for all families (original behavior)
      - list[str]: multiple roots used for all families (each family searched in each root)
      - dict[int, list[str]]: per-family roots, e.g.
          {
            1: [r"D:\Data\235\alarms", r"D:\Data\237\alarms"],
            2: [r"D:\Data\112\alarms", r"D:\Data\115\alarms", r"D:\Data\116\alarms"]
          }

    Expected directory under each root:
      <root>/processed-data/family{F}/*.hdf5
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
    #data_path = fr"D:\Data\{exp}\alarms"
    # data_path = fr"\\sanesstorage.cns.nyu.edu\archive\ginosar\Processed_data\Audio\{exp}" #'/mnt/home/mmartinez/ceph/data/gerbil/gily' # this directory contains .hdf5 files with spectrograms from your audio
    n_workers = max(os.cpu_count()-1,1) #len(os.sched_getaffinity(0))
    n_workers = 0
    print(n_workers)

    #(train_fns,test_fns),(train_ids,test_ids),specs_per_file = load_gerbils(data_path,specs_per_file=100,families=[1,2],test_size=0.2,seed=92,check=True)
    ### specs_per_file is how many spectrograms are in each .hdf5 file, families is the family number we're trying to load (I just set the data you sent to family 1),
    ### test_size is the portion of the data that will remain unseen in training, seed is used to maintain reproducibility, check determines whether we check to see if
    ### all files have 100 vocalization each

    # Combine families across experiments
    roots_per_family = {
        1: [r"D:\Data\235", r"D:\Data\237"],
        2: [r"D:\Data\112", r"D:\Data\113", r"D:\Data\114", r"D:\Data\115", r"D:\Data\116"],
    }
    out_dir = fr"D:\data\model_checkpoints"
    os.makedirs(out_dir, exist_ok=True)

    # (train_fns, test_fns), (train_ids, test_ids), specs_per_file = load_gerbils(
    #     gerbil_filepath=roots_per_family,  # dict: per-family roots
    #     specs_per_file=100,
    #     families=[1, 2],
    #     test_size=0.2,
    #     seed=92,
    #     check=True
    # )
    (train_fns, test_fns), (train_ids, test_ids), specs_per_file = load_gerbils_multi(
        gerbil_filepath=roots_per_family,  # dict: per-family roots
        specs_per_file=100,
        families=[1,2],
        test_size=0.2,
        seed=92,
        check=True
    )

    train_dataset = bird_data(train_fns, train_ids,specs_per_file=specs_per_file,transform=spec_to_tensor,conditional=False) # Conditional is false because we are not training with the arena info
    # train_dataset = bird_data(train_fns,train_ids,specs_per_file=specs_per_file,transform=lambda x: torch.from_numpy(x).to(torch.float32).unsqueeze(0), conditional=False)      # *GILY*: this is Mile's version, but I can't use lambda on multiprocessing in Windows

    ### Unfortunately, transform has to be a little weird because of how I saved the spectrograms. This performs these operations on each spectrogram before returning them
    ### Conditional determines if we want to condition our model on other variables (fm, entropy, syllable length, etc)

    test_dataset = bird_data(test_fns, test_ids,specs_per_file=specs_per_file,transform=spec_to_tensor, conditional=False)
    # test_dataset = bird_data(test_fns,test_ids,specs_per_file=specs_per_file,transform=lambda x: torch.from_numpy(x).to(torch.float32).unsqueeze(0), conditional=False)      # *GILY*: this is Mile's version, but I can't use lambda on multiprocessing in Windows

    train_loader = DataLoader(train_dataset,batch_size=64,num_workers=n_workers,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=64,num_workers=n_workers,shuffle=False)

    use_cuda = torch.cuda.is_available()
    pin = True if use_cuda else False

    from vocalizations.qmc_deep_gen.models.sampling import gen_fib_basis,gen_korobov_basis
    from vocalizations.qmc_deep_gen.models.utils import get_decoder_arch
    from vocalizations.qmc_deep_gen.models.qmc_base import QMCLVM
    import torch

    latent_dim=2 # sets our latent dimension
    ### If we use two dimensions, we should use gen_fib_basis for our grid over the latent space
    ### If more than two dimensions, we should use gen_korobov_basis. This requires additional arguments,
    ### if you want to use this see help(gen_korobov_basis) for good argument values

    latent_grid = gen_fib_basis(m=15) # m determines both the size of our grid and spacing of points
    ## if you want to plot this, you will need to plot (latent_grid % 1) instead of latent_grid


    dataset = 'gerbil_ava' # used for getting a pre-selected decoder architecture
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # use gpu if possible

    decoder = get_decoder_arch(dataset_name=dataset,latent_dim=latent_dim) # get_decoder_arch has a set of fixed architectures --
    ### if you want to play around with your own, you can make one using nn.Sequential (strings together layers). That's all that the
    ### decoders are -- nn.Sequential instances
    #%%
    from vocalizations.qmc_deep_gen.train.losses import binary_evidence,binary_lp,gaussian_evidence,gaussian_lp
    model = QMCLVM(latent_dim=latent_dim,device=device,decoder=decoder)

    ## binary evidence
    qmc_loss_func = binary_evidence # I used this for training models, but we can also use gaussian (what the VAE uses)
    qmc_lp = binary_lp

    from vocalizations.qmc_deep_gen.train.train import train_loop
    nEpochs=10

    #### to speed up training, you can decrease grid size (decrease m) at the expense of model performance,
    #### or increase batch size
    model, opt, losses = train_loop(
        model, train_loader, latent_grid.to(device), qmc_loss_func,
        nEpochs=nEpochs, verbose=True, conditional=False,
        out_dir=out_dir
    )
