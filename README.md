# Fake News Detection

Machine learning project to classify news articles/snippets as "Real" or "Fake" with a minimal Django web UI. It supports offline model training (TF‑IDF + Logistic Regression) and a one-page chat-like interface for real-time predictions.

***

## 🚀 Quickstart

1. Create/activate venv (PowerShell):

   ```bash
   .\virtualenv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Train the model (creates files under `models/`):

   ```bash
   python train_model.py
   ```

4. Run the web app:

   ```bash
   python newsdetect/manage.py migrate
   python newsdetect/manage.py runserver
   ```

5. Open the UI:

   - http://127.0.0.1:8000

Paste a news title/body and click Predict to see Fake/Real.

## 📝 Project Overview

This project addresses the need for rapid identification of misinformation. We built a text classification model that predicts whether a given piece of content is fake or real, deploying it through a simple web interface for immediate use by analysts.

The workflow is divided into four main stages:

1.  **Data Preparation**: We sourced a public dataset (e.g., ISOT Fake News Dataset), then cleaned and preprocessed the text by removing noise, handling missing values, and normalizing the content through tokenization and stopword removal.

2.  **Exploratory Data Analysis (EDA) & Feature Engineering**: We analyzed patterns in the text to distinguish between fake and real news, looking at features like article length, punctuation, and sentiment. We then converted the text into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization. 

3.  **Model Training & Validation**: Several classification models, such as **Logistic Regression** and **Naive Bayes**, were trained on the processed data. The models were rigorously evaluated using key metrics:
    * Accuracy
    * Precision
    * Recall
    * F1-Score
    A confusion matrix was also used to analyze the model's performance on false positives versus false negatives.

4.  **Deployment via Web Interface**: The best-performing model is exposed through a minimal **Django** web application with a single-page UI to paste a news snippet and get a "Fake" or "Real" prediction.

***

## 📁 Project Structure

```
GTC-Fake-News-Detection/
├─ ml_pipeline/               # Final Python ML pipeline (post-notebook experiments)
│  ├─ data.py                 # Load & clean raw CSVs, build cleaned dataset
│  ├─ preprocess.py           # Text normalization utilities
│  ├─ features.py             # Basic stats + TF‑IDF featurizer class
│  ├─ train.py                # Train + persist TF‑IDF + Logistic Regression
│  ├─ infer.py                # Load artifacts and run predictions
│  └─ __init__.py
├─ models/                    # Persisted artifacts
│  ├─ tfidf_vectorizer.joblib
│  └─ fake_news_lr.joblib
├─ newsdetect/                # Django web UI (one-page chat interface)
│  ├─ manage.py
│  ├─ newsdetect/
│  │  ├─ settings.py
│  │  ├─ urls.py
│  │  └─ wsgi.py
│  └─ frontend/
│     ├─ urls.py
│     ├─ views.py
│     └─ templates/frontend/chat.html
├─ Notebooks/                 # EDA, preprocessing, feature eng., modeling experiments
├─ train_model.py             # Offline training entrypoint
├─ requirements.txt
└─ README.md
```

- `ml_pipeline`: contains the final, productionized ML code derived from extensive notebook experimentation (EDA, feature engineering, and modeling), distilled into clean Python modules.
- `models`: trained artifacts (vectorizer and classifier) created by `train_model.py` / `ml_pipeline/train.py`.
- `newsdetect`: the Django project that hosts the simple web UI and prediction endpoint.
- `Notebooks`: Jupyter notebooks used during exploration and experimentation for EDA, preprocessing, feature engineering, and modeling.

***

## 🧪 Inference Flow (Web)

- UI (`GET /`) → `frontend.urls` → `chat_view` renders `templates/frontend/chat.html`.
- Predict (`POST /predict/`) → `predict_view` → loads `FakeNewsClassifier` → normalizes input → TF‑IDF transform → Logistic Regression prediction → JSON response.

***

## 👥 Team & Roles

| Name             | Role & Contributions                                                                                                 |
|------------------|-----------------------------------------------------------------------------------------------------------------------|
| Youssef Mohamed  | Converted notebooks to a clean Python ML pipeline; built Django UI/backend; Did CI/CD & deployment; created the presentation, README, and demo video. |
| Malak Badr       | Data Collection  Curated Kaggle dataset; merged ~45K articles; ensured class balance (~21K real, ~23K fake). |
| Mohammed Gaber   | Preprocessing  Deduplicated (−209), handled missing data, and implemented text preprocessing (lowercasing, URL/email/number removal, tokenization, stopwords/punctuation removal, lemmatization). |
| Mahmoud Mohamed  | EDA  Led analysis: class distribution (Fake > Real), length stats (avg ~405, max ~8K), wordclouds, top bigrams; confirmed weak length–label correlation. |
| Mayar El Hadi    | Feature Engineering  Built features: text stats, sentiment, TF‑IDF (5K), headline–text cosine similarity, MiniLM embeddings. |
| Maryam Hafez     | Modeling & Evaluation  Trained/evaluated models; baseline: LR 97%, NB 93%; tuned: LR 99% (best), NB 94%. |


***
