from langgraph.graph import StateGraph, END
from typing import TypedDict
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import joblib
import faiss
import numpy as np
import pickle
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

load_dotenv()

FEATURE_COLS = ["MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds"]

# Load everything once at module level, not inside each agent call
classifier = joblib.load("models/bbb_classifier.pkl")
# Compute training descriptor ranges directly from the same file the model trained on,
# so this is always accurate even if you retrain later with different data.
_train_df = pd.read_csv("data/processed/b3db_features.csv")
TRAIN_DESCRIPTOR_RANGES = {
    col: (_train_df[col].min(), _train_df[col].max())
    for col in ["MolWt", "LogP", "TPSA"]
}
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.read_index("faiss_index/literature.index")
with open("faiss_index/chunks.pkl", "rb") as f:
    corpus_chunks = pickle.load(f)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add it to the project .env file or your environment."
    )
gemini_client = genai.Client(api_key=gemini_api_key)

class MoleculeState(TypedDict):
    smiles: str
    valid: bool
    descriptors: dict
    prediction: float
    retrieved_context: list
    report: str
    error: str
    low_confidence: bool
    out_of_distribution: bool
    needs_extra_retrieval: bool

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

def validate_agent(state):
    mol = Chem.MolFromSmiles(state["smiles"])
    state["valid"] = mol is not None
    if not state["valid"]:
        state["error"] = "Invalid SMILES string"
    return state

def prediction_agent(state):
    if not state["valid"]:
        return state
    desc = get_descriptors(state["smiles"])
    X = pd.DataFrame([desc])[FEATURE_COLS]
    state["descriptors"] = desc
    state["prediction"] = float(classifier.predict_proba(X)[0][1])
    return state

def retrieval_agent(state, k=3):
    if not state["valid"]:
        return state
    d = state["descriptors"]
    query = f"molecule with TPSA {d['TPSA']:.1f}, LogP {d['LogP']:.1f}, molecular weight {d['MolWt']:.1f}"
    q_embedding = embed_model.encode([query])
    _, indices = faiss_index.search(np.array(q_embedding).astype("float32"), k=k)
    state["retrieved_context"] = [corpus_chunks[i] for i in indices[0]]
    return state

def report_agent(state):
    if not state["valid"]:
        state["report"] = f"Could not process molecule: {state['error']}"
        return state

    confidence_note = ""
    if state.get("low_confidence"):
        confidence_note = "Note: this prediction has low confidence and should be treated as uncertain. "
    if state.get("out_of_distribution"):
        confidence_note += "This molecule falls outside the descriptor ranges seen during training, so extrapolation risk is higher."

    desc = state["descriptors"]
    context = "\n\n".join(state["retrieved_context"])

    prompt = f"""You are explaining a machine learning model's prediction about blood-brain barrier (BBB) permeability to a researcher.

Molecule descriptors:
- Molecular weight: {desc['MolWt']:.1f}
- LogP: {desc['LogP']:.2f}
- TPSA: {desc['TPSA']:.1f}
- H-bond donors: {desc['NumHDonors']}
- H-bond acceptors: {desc['NumHAcceptors']}
- Rotatable bonds: {desc['NumRotatableBonds']}

Model prediction: {state['prediction']:.3f} probability of BBB permeability (0 = non-permeable, 1 = permeable)

{confidence_note}

Relevant literature excerpts:
{context}

Write a short, clear explanation (3-5 sentences) of this prediction. State the prediction plainly, then explain which descriptor(s) most likely drove it, citing the literature excerpts above where they support your explanation. If a confidence note is present above, mention it explicitly near the start of your explanation, not buried at the end. Do not invent facts not supported by the descriptors or the literature excerpts."""

    response = gemini_client.models.generate_content(
        model="gemini-3.7-flash",
        contents=prompt,
    )
    state["report"] = response.text
    return state

def verifier_agent(state):
    if not state["valid"]:
        return state

    confidence = abs(state["prediction"] - 0.5) * 2  # 0 = totally unsure, 1 = totally sure
    desc = state["descriptors"]

    out_of_range = any(
        not (TRAIN_DESCRIPTOR_RANGES[k][0] <= desc[k] <= TRAIN_DESCRIPTOR_RANGES[k][1])
        for k in TRAIN_DESCRIPTOR_RANGES
    )

    state["low_confidence"] = confidence < 0.3
    state["out_of_distribution"] = out_of_range
    state["needs_extra_retrieval"] = state["low_confidence"] or state["out_of_distribution"]
    return state

def route_after_verify(state):
    if not state["valid"]:
        return "report"
    return "retrieve_more" if state["needs_extra_retrieval"] else "retrieve"

def retrieve_more_agent(state):
    return retrieval_agent(state, k=6)  # pull more context when the model is unsure

def build_graph():
    graph = StateGraph(MoleculeState)
    graph.add_node("validate", validate_agent)
    graph.add_node("predict", prediction_agent)
    graph.add_node("verify", verifier_agent)
    graph.add_node("retrieve", retrieval_agent)
    graph.add_node("retrieve_more", retrieve_more_agent)
    graph.add_node("report", report_agent)
    graph.add_edge("validate", "predict")
    graph.add_edge("predict", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {
        "retrieve": "retrieve",
        "retrieve_more": "retrieve_more",
        "report": "report",  # invalid SMILES skips straight to report
    })
    graph.add_edge("retrieve", "report")
    graph.add_edge("retrieve_more", "report")
    graph.add_edge("report", END)
    graph.set_entry_point("validate")
    return graph.compile()

app = build_graph()

if __name__ == "__main__":
    test_cases = {
        "normal (caffeine)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "extreme (constructed high-MW molecule)": "CCCCCCCCCCCCCCCCCCCCC(=O)OCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
        "invalid smiles": "not_a_real_smiles_string!!",
    }

    for label, smiles in test_cases.items():
        print(f"\n--- {label} ---")
        result = app.invoke({"smiles": smiles})
        print("valid:", result.get("valid"))
        print("prediction:", result.get("prediction"))
        print("low_confidence:", result.get("low_confidence"))
        print("out_of_distribution:", result.get("out_of_distribution"))
        print("needs_extra_retrieval:", result.get("needs_extra_retrieval"))
        print("num retrieved chunks:", len(result.get("retrieved_context", [])))