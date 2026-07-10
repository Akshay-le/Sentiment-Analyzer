# Sentiment Analyzer — Movie Reviews

A complete, deployable sentiment analysis web app for movie reviews.
TF-IDF + Logistic Regression model trained on the 50,000-review IMDB
dataset (~91% test accuracy), served with Flask, and a custom
cinema-marquee themed UI (plain HTML/CSS/JS — no Streamlit, no
frontend framework required).

## What's inside

```
sentiment-app/
├── app.py                 # Flask app + /api/predict endpoint
├── train_model.py         # Trains & saves the model (already run — model/ is included)
├── model/
│   ├── sentiment_model.pkl
│   ├── vectorizer.pkl
│   └── metrics.json
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── requirements.txt
├── Procfile                # for Heroku / Render
├── Dockerfile               # for Docker / Railway / Fly.io / Cloud Run
└── README.md
```

## Run it locally

```bash
cd sentiment-app
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# (Optional) retrain the model — a trained model is already committed
# python3 train_model.py

python3 app.py
```

Open **http://127.0.0.1:5000** in your browser.

For production locally:
```bash
gunicorn app:app --bind 0.0.0.0:5000 --workers 2
```

## How it works

- **Model**: `TfidfVectorizer` (unigrams + bigrams, 40k features) feeding a
  `LogisticRegression` classifier. Trained on 40,000 reviews, tested on
  10,000 held-out reviews → **90.6% accuracy / 0.907 F1**.
- **API**: `POST /api/predict` with `{"review": "..."}` returns the
  predicted label, confidence, positive/negative probability, and the
  top words (from the review) that most influenced the verdict —
  a simple, honest explainability layer using the model's own
  coefficients (no black box).
- **UI**: no template frameworks — hand-built HTML/CSS/JS styled as a
  cinema marquee / ticket stub, fully responsive, with a loading state,
  error states, and a "try a sample" button.

## Retraining on your own data

`train_model.py` expects a CSV with `review` and `sentiment` columns
(`sentiment` = `positive`/`negative`). Point `DATA_PATH` at your file and
re-run — it will overwrite `model/sentiment_model.pkl` and
`model/vectorizer.pkl`.

## Deployment options

### Render / Railway (easiest)
1. Push this folder to a GitHub repo.
2. Create a new **Web Service**, connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
   (Render/Railway auto-detect this from the included `Procfile`.)

### Heroku
```bash
heroku create your-app-name
git push heroku main
```
The included `Procfile` handles the start command.

### Docker (any host: Fly.io, Cloud Run, AWS, a VPS, etc.)
```bash
docker build -t sentiment-app .
docker run -p 5000:5000 sentiment-app
```

### Notes for production
- `model/` (~2 MB total) is small enough to commit directly to git — no
  external storage or Git LFS needed.
- The app is stateless — you can run any number of instances behind a
  load balancer with no shared state required.
- `app.py` reads `PORT` from the environment, so it works out of the box
  on Render/Heroku/Railway, which inject `$PORT` automatically.
