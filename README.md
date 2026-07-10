# Sentiment Analyzer — Movie Reviews

## What this project is

A full-stack web application that reads a movie review typed in by a
user and predicts whether the sentiment is **positive** or **negative**,
along with a confidence score and the specific words in the review that
influenced the decision most.

It's a complete, self-contained machine learning project — not a demo
wired to an external API. The model is trained from scratch on a real
dataset, saved to disk, and served through a custom backend and
frontend that were both built for this project.

## What it does

1. User types or pastes a movie review into the web interface.
2. The review is sent to a backend API.
3. A trained machine learning model classifies it as positive or
   negative and returns a confidence percentage.
4. The interface displays the verdict, a confidence bar, a
   positive/negative probability split, and the specific words from
   that review that pushed the model's decision one way or the other.

## Technologies used

| Layer | Technology |
|---|---|
| Language | Python 3 (backend, model training), JavaScript (frontend logic) |
| Machine learning | scikit-learn (`TfidfVectorizer`, `LogisticRegression`) |
| Data handling | pandas |
| Model serialization | joblib |
| Backend framework | Flask (Python) |
| Production server | Gunicorn |
| Frontend | Hand-written HTML5, CSS3, vanilla JavaScript — no frontend framework, no Streamlit |
| Templating | Jinja2 (Flask's built-in template engine, used in `index.html`) |
| Deployment | Docker, with support for Render, Railway, and Heroku |



## How the model was created and trained

**1. Dataset**
The model is trained on the IMDB Movie Reviews dataset — 50,000 real
movie reviews collected from IMDB, evenly split between 25,000
positive and 25,000 negative examples. This is one of the standard,
widely-used academic datasets for sentiment analysis.

**2. Text cleaning (preprocessing)**
Each raw review goes through a cleaning step before it's usable:
- HTML tags (like `<br />`, which appear in raw IMDB text) are stripped out
- Text is lowercased
- Punctuation and numbers are removed, leaving only words
- Extra whitespace is collapsed

**3. Feature extraction — TF-IDF**
Cleaned text can't be fed directly into a machine learning model — it
has to become numbers. This project uses **TF-IDF (Term
Frequency–Inverse Document Frequency)** vectorization, which converts
each review into a numeric vector representing how important each word
(and pair of adjacent words — bigrams) is to that specific review
relative to the whole dataset. The vectorizer keeps the 40,000 most
informative word/phrase features and ignores common English stop words
(like "the," "and," "is").

**4. Splitting the data**
The 50,000 reviews are split into:
- 40,000 reviews for **training** the model
- 10,000 reviews held out for **testing** — the model never sees these
  during training, so they give an honest measure of real-world accuracy

**5. Training the classifier**
A **Logistic Regression** model is trained on the TF-IDF vectors from
the training set. Logistic Regression is a fast, well-understood
linear classifier that works especially well on high-dimensional
sparse text data like TF-IDF vectors, and — unlike more opaque models —
its learned weights can be directly inspected, which is what powers
the "words that swayed the verdict" feature in the UI.

**6. Evaluating the model**
Once trained, the model is run against the 10,000 held-out test
reviews it has never seen, and scored:

- **Accuracy: 90.6%**
- **F1 Score: 0.907**

These numbers are saved to `model/metrics.json` and represent genuine
performance on unseen data, not just performance on the training set.

**7. Saving the model**
Two artifacts are saved to disk with `joblib`:
- `vectorizer.pkl` — the fitted TF-IDF vectorizer (so new reviews can
  be converted into the same numeric format the model expects)
- `sentiment_model.pkl` — the trained Logistic Regression classifier

These are loaded once when the Flask server starts, so predictions at
request time are just a vectorize-and-predict call — fast enough for
real-time use, with no retraining needed unless you want to.

**8. Explainability**
Because Logistic Regression assigns a learned weight to every word
feature, the app looks up which words in a given review have the
strongest weights (positive or negative) and surfaces the top ones in
the UI. This isn't a separate "explainability model" — it's reading
the same weights the classifier already uses to make its decision.





are denser or more strongly weighted — a confidence score below ~80%
is the model's own signal that a review is ambiguous.
