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
import time
from sentence_transformers import SentenceTransformer
from google import genai

FEATURE_COLS = ["MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds"]

# Load everything once at module level, not inside each agent call
classifier = joblib.load("models/bbb_classifier.pkl")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.read_index("faiss_index/literature.index")
with open("faiss_index/chunks.pkl", "rb") as f:
    corpus_chunks = pickle.load(f)
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

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
    report_degraded: bool

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

def build_fallback_report(state, confidence_note):
    """Rule-based report used when Gemini is unavailable after retries.
    No LLM call — just formats what the pipeline already computed, so the
    app still returns something useful instead of failing outright."""
    desc = state["descriptors"]
    verdict = "likely permeable" if state["prediction"] >= 0.5 else "likely non-permeable"
    lines = [
        f"[Automated summary — AI explanation service temporarily unavailable]",
        f"Prediction: {verdict} (model score: {state['prediction']:.3f}).",
        f"Descriptors: MolWt={desc['MolWt']:.1f}, LogP={desc['LogP']:.2f}, "
        f"TPSA={desc['TPSA']:.1f}, HBD={desc['NumHDonors']}, HBA={desc['NumHAcceptors']}.",
    ]
    if confidence_note:
        lines.append(confidence_note.strip())
    if state["retrieved_context"]:
        lines.append(f"{len(state['retrieved_context'])} related literature passages were retrieved but could not be summarized right now.")
    return "\n".join(lines)

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

    # Retry with exponential backoff for transient errors (503 UNAVAILABLE, rate limits, etc.)
    # After 3 failed attempts, fall back to a rule-based report instead of crashing the request.
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            state["report"] = response.text
            state["report_degraded"] = False
            return state
        except Exception as e:
            error_str = str(e)
            is_transient = "503" in error_str or "UNAVAILABLE" in error_str or "429" in error_str
            if is_transient and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 1.5  # 1.5s, 3s, 6s
                print(f"Gemini call failed (attempt {attempt+1}/{max_retries}): {error_str}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"Gemini call failed permanently after {attempt+1} attempts: {error_str}")
                state["report"] = build_fallback_report(state, confidence_note)
                state["report_degraded"] = True
                return state

    return state

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
    # Quick standalone test before wiring into Flask
    result = app.invoke({"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"})  # caffeine
    print(result)