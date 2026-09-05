import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # so it can import agents/

from flask import Flask, request, jsonify
from flask_cors import CORS
from agents.pipeline import app as agent_pipeline

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    smiles = data.get("smiles", "")
    if not smiles:
        return jsonify({"error": "No SMILES provided"}), 400
    try:
        result = agent_pipeline.invoke({"smiles": smiles})
        return jsonify(result)
    except Exception as error:
        return jsonify({"error": str(error)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)