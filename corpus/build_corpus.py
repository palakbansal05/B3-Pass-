from Bio import Entrez
import pickle

Entrez.email = "palakkb.05@gmail.com"  # required by NCBI, use your real email

def fetch_abstracts(query, max_results=200):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    ids = Entrez.read(handle)["IdList"]
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
    return handle.read()

def main():
    queries = [
        "blood brain barrier permeability descriptors",
        "CNS MPO score drug design",
        "Lipinski rule of five CNS penetration",
        "TPSA LogP blood brain barrier",
    ]
    all_text = []
    for q in queries:
        print(f"Fetching: {q}")
        all_text.append(fetch_abstracts(q))

    with open("corpus/raw_abstracts.pkl", "wb") as f:
        pickle.dump(all_text, f)
    print(f"Saved {len(all_text)} query results to corpus/raw_abstracts.pkl")

if __name__ == "__main__":
    main()