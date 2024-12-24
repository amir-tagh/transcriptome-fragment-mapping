import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm


def calculate_tversky(fp1, fp2, alpha=0.5, beta=0.5):
    """Calculate Tversky similarity between two fingerprints."""
    common = DataStructs.FingerprintSimilarity(fp1, fp2)  # Intersection count
    a_only = DataStructs.FingerprintSimilarity(fp1, ~fp2)  # Unique to fp1
    b_only = DataStructs.FingerprintSimilarity(fp2, ~fp1)  # Unique to fp2

    # Tversky index formula
    return common / (common + alpha * a_only + beta * b_only)


def get_fingerprints(smiles_list, method):
    """Generate fingerprints for a list of SMILES strings."""
    fingerprints = []
    for smiles in tqdm(smiles_list, desc="Generating fingerprints"):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            if method == "morgan":
                # Morgan (Circular) fingerprint
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            elif method == "rdkit":
                # RDKit (Topological) fingerprint
                fp = Chem.RDKFingerprint(mol)
            else:
                raise ValueError("Invalid fingerprint method! Use 'morgan' or 'rdkit'.")
            fingerprints.append(fp)
        else:
            fingerprints.append(None)  # Placeholder for invalid molecules
    return fingerprints


def screen_smiles(smiles1, smiles2, cutoff, alpha, beta, method):
    """Screen SMILES from the first list against the second list."""
    fingerprints1 = get_fingerprints(smiles1, method)
    fingerprints2 = get_fingerprints(smiles2, method)

    # Filter based on Tversky similarity
    selected_smiles = []
    for i, fp1 in enumerate(fingerprints1):
        if fp1 is None:
            continue  # Skip invalid entries
        for fp2 in fingerprints2:
            if fp2 is None:
                continue
            similarity = calculate_tversky(fp1, fp2, alpha, beta)
            if similarity >= cutoff:
                selected_smiles.append(smiles1[i])
                break  # Move to the next molecule after first match
    return selected_smiles


def main():
    """Main function to handle arguments and execute the script."""
    parser = argparse.ArgumentParser(description="Screen SMILES lists using Tversky similarity.")
    parser.add_argument('list1', help='First input file (tab-separated) containing SMILES.')
    parser.add_argument('list2', help='Second input file (tab-separated) containing SMILES.')
    parser.add_argument('--output', required=True, help='Output file for selected SMILES.')
    parser.add_argument('--cutoff', type=float, default=0.7, help='Similarity cutoff (default: 0.7).')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter for Tversky index (default: 0.5).')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta parameter for Tversky index (default: 0.5).')
    parser.add_argument('--method', choices=['morgan', 'rdkit'], default='morgan',
                        help="Fingerprint type: 'morgan' (default) or 'rdkit'.")

    args = parser.parse_args()

    # Read input files
    smiles1 = pd.read_csv(args.list1, sep="\t", header=None)[0].tolist()
    smiles2 = pd.read_csv(args.list2, sep="\t", header=None)[0].tolist()

    # Screen molecules
    selected_smiles = screen_smiles(smiles1, smiles2, args.cutoff, args.alpha, args.beta, args.method)

    # Save output
    pd.DataFrame(selected_smiles).to_csv(args.output, sep="\t", index=False, header=False)
    print(f"Saved {len(selected_smiles)} selected molecules to {args.output}")


if __name__ == "__main__":
    main()

