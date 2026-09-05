from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open("corpus/raw_abstracts.pkl", "rb") as f:
        raw_texts = pickle.load(f)
    with open("corpus/manual_papers.txt", "r", encoding="utf-8") as f:
        manual_text = f.read()

    all_chunks = []
    for text in raw_texts:
        all_chunks.extend(chunk_text(text))
    all_chunks.extend(chunk_text(manual_text))

    print(f"Total chunks: {len(all_chunks)}")

    embeddings = model.encode(all_chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, "faiss_index/literature.index")

    with open("faiss_index/chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    print("Saved FAISS index and chunk store.")

if __name__ == "__main__":
    main()