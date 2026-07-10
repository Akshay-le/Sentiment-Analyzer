"""
Train a Logistic Regression sentiment classifier on the IMDB movie
reviews dataset (50,000 reviews) and persist the TF-IDF vectorizer
and model to disk for the Flask app to load at request time.
"""
import re
import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

DATA_PATH = "data_raw.csv"
MODEL_PATH = "model/sentiment_model.pkl"
VEC_PATH = "model/vectorizer.pkl"
META_PATH = "model/metrics.json"

TAG_RE = re.compile(r"<.*?>")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = TAG_RE.sub(" ", text)          # strip HTML tags e.g. <br />
    text = text.lower()
    text = NON_ALPHA_RE.sub(" ", text)    # strip punctuation / numbers
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()
    print(f"Rows: {len(df)}")

    print("Cleaning text...")
    df["clean_review"] = df["review"].astype(str).apply(clean_text)
    df["label"] = (df["sentiment"] == "positive").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_review"], df["label"], test_size=0.2,
        random_state=42, stratify=df["label"]
    )

    print("Vectorizing (TF-IDF, unigrams+bigrams, 40k features)...")
    vectorizer = TfidfVectorizer(
        max_features=40000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression...")
    t0 = time.time()
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear")
    clf.fit(X_train_vec, y_train)
    print(f"Trained in {time.time() - t0:.1f}s")

    preds = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}  F1: {f1:.4f}")
    print(classification_report(y_test, preds, target_names=["negative", "positive"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    import os, json
    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)
    with open(META_PATH, "w") as f:
        json.dump({
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "features": X_train_vec.shape[1],
        }, f, indent=2)
    print(f"Saved model to {MODEL_PATH} and vectorizer to {VEC_PATH}")


if __name__ == "__main__":
    main()
