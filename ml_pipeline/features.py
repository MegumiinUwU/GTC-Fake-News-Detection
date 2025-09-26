import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def add_basic_stats(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:

    stats = pd.DataFrame(index=df.index)
    texts = df[text_col].astype(str)
    stats["text_length"] = texts.apply(lambda x: len(x.split()))
    stats["unique_words"] = texts.apply(lambda x: len(set(x.split())))
    stats["unique_ratio"] = (stats["unique_words"] / stats["text_length"]).replace([np.inf, -np.inf], 0).fillna(0)
    return pd.concat([df.reset_index(drop=True), stats.reset_index(drop=True)], axis=1)


class TfidfFeaturizer:

    def __init__(self, max_features: int = 5000, ngram_range: tuple[int, int] = (1, 2)) -> None:

        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words="english")

    def fit(self, texts: pd.Series) -> "TfidfFeaturizer":

        self.vectorizer.fit(texts.astype(str))
        return self

    def transform(self, texts: pd.Series):

        return self.vectorizer.transform(texts.astype(str))

    def fit_transform(self, texts: pd.Series):

        return self.vectorizer.fit_transform(texts.astype(str))


