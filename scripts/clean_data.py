from rdkit import Chem
import pandas as pd

def canonicalize(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None

def main():
    b3db = pd.read_csv("data/raw/B3DB_classification.tsv", sep="\t")
    bbbp = pd.read_csv("data/raw/BBBP.csv")

    # --- B3DB: SMILES is fully populated; this dropna is a defensive check, not a real fix ---
    b3db = b3db.dropna(subset=["SMILES"])
    b3db["canonical_smiles"] = b3db["SMILES"].apply(canonicalize)
    b3db["label"] = b3db["BBB+/BBB-"].map({"BBB+": 1, "BBB-": 0})
    b3db = b3db.dropna(subset=["canonical_smiles", "label"])

    # --- BBBP: already binary via p_np, just canonicalize ---
    bbbp["canonical_smiles"] = bbbp["smiles"].apply(canonicalize)
    bbbp = bbbp.dropna(subset=["canonical_smiles"])
    bbbp["label"] = bbbp["p_np"]  # already 0/1

    assert set(b3db["label"].unique()) <= {0, 1}
    assert set(bbbp["label"].unique()) <= {0, 1}

    # Remove any BBBP molecules that also exist in B3DB — keeps the test set clean
    overlap = set(b3db["canonical_smiles"]) & set(bbbp["canonical_smiles"])
    bbbp_clean = bbbp[~bbbp["canonical_smiles"].isin(overlap)]

    print(f"B3DB rows after cleaning: {len(b3db)}")
    print(f"Removed {len(overlap)} overlapping molecules from BBBP test set")
    print(f"BBBP test rows after dedup: {len(bbbp_clean)}")

    b3db.to_csv("data/processed/b3db_clean.csv", index=False)
    bbbp_clean.to_csv("data/processed/bbbp_clean.csv", index=False)

if __name__ == "__main__":
    main()