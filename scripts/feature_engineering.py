import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
    }

def add_features(df, smiles_col="canonical_smiles"):
    features = df[smiles_col].apply(get_descriptors).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), features], axis=1)

def main():
    b3db = pd.read_csv("data/processed/b3db_clean.csv")
    bbbp = pd.read_csv("data/processed/bbbp_clean.csv")

    b3db_features = add_features(b3db)
    bbbp_features = add_features(bbbp)

    b3db_features.to_csv("data/processed/b3db_features.csv", index=False)
    bbbp_features.to_csv("data/processed/bbbp_features.csv", index=False)
    print("Saved feature files for both datasets.")

if __name__ == "__main__":
    main()