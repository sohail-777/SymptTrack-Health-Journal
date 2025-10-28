# app.py
import time
import torch
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
CORS(app)

MODEL_NAME = "bvanaken/CORe-clinical-diagnosis-prediction"
print(f"[app] Loading tokenizer & model {MODEL_NAME} (may take time)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
print("[app] Model loaded.")


def predict_icd9_codes(text, threshold=0.3, top_k=10):

    if not text:
        return []
    tokenized = tokenizer(text, return_tensors="pt",
                          truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        out = model(**tokenized)
    logits = out.logits
    probs = torch.sigmoid(logits)[0].cpu().tolist()
    preds = []

    for idx, p in enumerate(probs):
        if p >= threshold:
            label = model.config.id2label.get(idx, str(idx))
            preds.append({"label": label, "prob": float(p)})

    preds = sorted(preds, key=lambda x: x["prob"], reverse=True)[:top_k]
    return preds


@app.route("/")
def route_login():
    return render_template("login.html")


@app.route("/home")
def route_home():
    return render_template("home.html")


@app.route("/symptoms")
def route_symptoms():
    return render_template("symptoms.html")


@app.route("/records")
def route_records():
    return render_template("records.html")


@app.route("/about")
def route_about():
    return render_template("about.html")


@app.route("/diagnosis", methods=["POST"])
def route_diagnosis():

    payload = request.get_json(force=True)

    text = payload.get("text") or payload.get("symptoms") or ""
    date = payload.get("date") or payload.get("dateIso") or payload.get(
        "date_iso") or time.strftime("%Y-%m-%d")

    if not text or len(text.strip()) == 0:
        return jsonify({"error": "No symptom text provided"}), 400

    try:
        diagnoses = predict_icd9_codes(text, threshold=0.3)
    except Exception as e:

        diagnoses = []

    return jsonify({
        "diagnoses": diagnoses,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
