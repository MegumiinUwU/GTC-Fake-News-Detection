import os
import joblib
from .preprocess import normalize_text


class FakeNewsClassifier:

    def __init__(self, model_dir: str = "models") -> None:

        self.model_path = os.path.join(model_dir, "fake_news_lr.joblib")
        self.vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

    def predict_label(self, title: str, text: str) -> int:

        combined = f"{title or ''} {text or ''}".strip()
        combined_norm = normalize_text(combined)
        features = self.vectorizer.transform([combined_norm])
        pred = self.model.predict(features)[0]
        return int(pred)

    def predict_proba(self, title: str, text: str) -> float:

        combined = f"{title or ''} {text or ''}".strip()
        combined_norm = normalize_text(combined)
        features = self.vectorizer.transform([combined_norm])
        proba = self.model.predict_proba(features)[0][1]
        return float(proba)


