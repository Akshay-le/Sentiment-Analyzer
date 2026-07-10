import os
import re
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

TAG_RE = re.compile(r"<.*?>")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = TAG_RE.sub(" ", text)
    text = text.lower()
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


# ---- Load model artifacts once at startup ----
MODEL_PATH = os.path.join(BASE_DIR, "model", "sentiment_model.pkl")
VEC_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")
META_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

metrics = {}
if os.path.exists(META_PATH):
    with open(META_PATH) as f:
        metrics = json.load(f)

feature_names = np.array(vectorizer.get_feature_names_out())
coefs = model.coef_[0]


def top_contributing_words(clean, top_n=8):
    """Return the words in the review that most pushed the prediction
    toward positive / negative, for a simple explainability display."""
    tokens = set(clean.split())
    contributions = []
    for tok in tokens:
        idx = np.where(feature_names == tok)[0]
        if idx.size:
            contributions.append((tok, float(coefs[idx[0]])))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions[:top_n]


@app.route("/")
def home():
    return render_template("index.html", metrics=metrics)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    review = (data.get("review") or "").strip()

    if not review:
        return jsonify({"error": "Please enter a movie review."}), 400
    if len(review) < 3:
        return jsonify({"error": "Review is too short to analyze."}), 400

    clean = clean_text(review)
    if not clean:
        return jsonify({"error": "Review doesn't contain analyzable text."}), 400

    vec = vectorizer.transform([clean])
    proba = model.predict_proba(vec)[0]  # [neg, pos]
    pred = model.predict(vec)[0]

    label = "positive" if pred == 1 else "negative"
    confidence = float(proba[1] if pred == 1 else proba[0])

    words = top_contributing_words(clean)
    top_words = [
        {"word": w, "weight": round(wt, 3), "direction": "positive" if wt > 0 else "negative"}
        for w, wt in words
    ]

    return jsonify({
        "label": label,
        "confidence": round(confidence * 100, 1),
        "positive_prob": round(float(proba[1]) * 100, 1),
        "negative_prob": round(float(proba[0]) * 100, 1),
        "top_words": top_words,
    })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "metrics": metrics})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
