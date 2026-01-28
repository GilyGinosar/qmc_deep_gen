import h5py
import matplotlib.pyplot as plt
import numpy as np
import math
import os


def plot_and_save_hdf5_inventory(hdf5_path, dataset_key='specs', output_folder=None):
    """
    Plots every spectrogram in an HDF5 file and saves the result as a PNG.
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Load all data
        specs = f[dataset_key][:]
        onsets = f['onsets'][:]
        offsets = f['offsets'][:]
        filenames = [fn.decode('utf-8') for fn in f['audio_filenames'][:]]
        locations = [loc.decode('utf-8') for loc in f['locations'][:]]

        num_specs = len(specs)
        cols = 5
        rows = math.ceil(num_specs / cols)

        # Dynamically scale height: 3 inches per row
        fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3))
        axes = axes.flatten()

        for i in range(num_specs):
            ax = axes[i]
            # Use 'magma' to match your diagnostic plots
            ax.imshow(specs[i], origin='lower', aspect='auto', cmap='magma')

            # Format metadata for title
            file_base = os.path.basename(filenames[i])
            title = (f"#{i} | {locations[i]}\n{file_base}\n{onsets[i]:.2f}s-{offsets[i]:.2f}s")

            ax.set_title(title, fontsize=7)
            ax.axis('off')

        # Hide any unused subplots in the grid
        for j in range(num_specs, len(axes)):
            axes[j].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(f"Inventory: {dataset_key} | File: {os.path.basename(hdf5_path)}", fontsize=16)

        # -- Save Logic --
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            save_name = os.path.basename(hdf5_path).replace('.hdf5', f'_{dataset_key}_inventory.png')
            save_path = os.path.join(output_folder, save_name)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')  # High DPI for clarity
            print(f"Saved inventory plot to: {save_path}")

        plt.show()


# Example Usage:
path_to_plot = r"\\sanesstorage.cns.nyu.edu\archive\ginosar\Processed_data\Audio\237\processed-data\family1_25_1_H\syllables_0002.hdf5"

# 1. Get the directory containing the HDF5 file
base_dir = os.path.dirname(path_to_plot) # This removes 'syllables_0002.hdf5'

# 2. Correctly join the path strings
output_path = os.path.join(base_dir, 'inventories')

# 3. Call your function
plot_and_save_hdf5_inventory(path_to_plot, output_folder=output_path)
