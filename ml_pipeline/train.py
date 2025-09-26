import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from .preprocess import normalize_series
from .features import TfidfFeaturizer


def train_and_save(
    cleaned_csv_path: str,
    model_dir: str = "models",
    text_cols: tuple[str, str] = ("title", "text"),
    label_col: str = "label",
) -> dict:

    df = pd.read_csv(cleaned_csv_path)
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    df["combined_text"] = (df[text_cols[0]] + " " + df[text_cols[1]]).str.strip()
    df = df[df["combined_text"] != ""].reset_index(drop=True)

    df["combined_text_norm"] = normalize_series(df["combined_text"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["combined_text_norm"], df[label_col], test_size=0.2, random_state=42, stratify=df[label_col]
    )

    featurizer = TfidfFeaturizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = featurizer.fit_transform(X_train)
    X_test_tfidf = featurizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, solver="liblinear", C=100, penalty="l1", random_state=42)
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, "fake_news_lr.joblib"))
    joblib.dump(featurizer.vectorizer, os.path.join(model_dir, "tfidf_vectorizer.joblib"))

    return {"accuracy": acc, "report": report}


if __name__ == "__main__":
    cleaned_csv = "cleaned_fake_news.csv"
    if not os.path.exists(cleaned_csv):
        from .data import build_clean_dataset

        build_clean_dataset("True.csv", "Fake.csv", cleaned_csv)

    metrics = train_and_save(cleaned_csv)
    print({"accuracy": round(metrics["accuracy"], 4)})


