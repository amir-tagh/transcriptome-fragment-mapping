import argparse
from rdkit import Chem
from rdkit.Chem import AllChem, RDKFingerprint
from rdkit import DataStructs
import numpy as np
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar

def read_smiles(file_path):
    """Read SMILES strings from a file."""
    with open(file_path, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    return smiles_list

def calculate_morgan_fps(smiles_list, radius=2, n_bits=1024):
    """Calculate Morgan fingerprints for a list of SMILES with a progress bar."""
    fps = []
    for smi in tqdm(smiles_list, desc="Calculating Morgan fingerprints"):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
    return np.array(fps)

def calculate_rdkit_fps(smiles_list, n_bits=1024):
    """Calculate RDKit topological fingerprints for a list of SMILES with a progress bar."""
    fps = []
    for smi in tqdm(smiles_list, desc="Calculating RDKit fingerprints"):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = RDKFingerprint(mol, fpSize=n_bits)
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
    return np.array(fps)

def run_umap(fps, n_neighbors=15, min_dist=0.1, n_components=2):
    """Run UMAP on the fingerprint data with a progress bar."""
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    embedding = reducer.fit_transform(fps)
    return embedding

import matplotlib.pyplot as plt

def plot_umap(embeddings_list, labels_list, title="UMAP Overlap", save_path=None):
    """
    Plot UMAP embeddings with overlapping scatter plots and save the figure.

    Parameters:
        embeddings_list (list of np.ndarray): A list of UMAP embeddings (2D arrays).
        labels_list (list of str): A list of labels for each embedding set.
        title (str): The title of the plot.
        save_path (str): Path to save the figure (optional).

    Returns:
        None
    """
    plt.figure(figsize=(10, 8))
    ax = plt.gca()  # Get the current axes
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']  # Preset colors

    # Plot UMAP embeddings
    for idx, (embedding, label) in enumerate(zip(embeddings_list, labels_list)):
        color = colors[idx % len(colors)]  # Cycle through colors
        ax.scatter(
            embedding[:, 0], embedding[:, 1],
            label=label, alpha=0.6, s=10, c=color, edgecolor='none'
        )

    # Customize spines
    ax.spines[['right', 'top']].set_visible(False)  # Hide top and right spines
    ax.spines["left"].set_linewidth(1.5)  # Thicker left spine
    ax.spines["bottom"].set_linewidth(1.5)  # Thicker bottom spine

    # Customize tick marks
    ax.tick_params(axis="x", width=2)  # Make x-tick marks thicker
    ax.tick_params(axis="y", width=2)  # Make y-tick marks thicker

    # Enhance plot details
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=14, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.legend(loc="best", fontsize=10)

    # Layout adjustment
    plt.tight_layout()

    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    # Display the plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Calculate UMAP for multiple SMILES files using Morgan or RDKit fingerprints.")
    parser.add_argument('files', nargs='+', help="Paths to files containing SMILES strings.")
    parser.add_argument('--fingerprint', choices=['morgan', 'rdkit'], default='morgan',
                        help="Type of fingerprint to use (morgan or rdkit). Default is Morgan.")
    parser.add_argument('--radius', type=int, default=2, help="Radius for Morgan fingerprints (default: 2).")
    parser.add_argument('--n_bits', type=int, default=1024, help="Number of bits in the fingerprint vector (default: 1024).")
    parser.add_argument('--output', type=str, default="umap_plot.png",
                        help="Path to save the UMAP plot (default: 'umap_plot.png').")
    args = parser.parse_args()

    embeddings_list = []
    labels_list = []

    # Process each SMILES file
    for file in args.files:
        print(f"Processing file: {file}")
        smiles_list = read_smiles(file)
        
        if args.fingerprint == 'morgan':
            fps = calculate_morgan_fps(smiles_list, radius=args.radius, n_bits=args.n_bits)
        else:
            fps = calculate_rdkit_fps(smiles_list, n_bits=args.n_bits)
        
        embedding = run_umap(fps)
        embeddings_list.append(embedding)
        labels_list.append(file)

    # Plot the overlapping UMAP embeddings and save the figure
    plot_umap(embeddings_list, labels_list, title="UMAP of SMILES Files", save_path=args.output)

if __name__ == "__main__":
    main()

