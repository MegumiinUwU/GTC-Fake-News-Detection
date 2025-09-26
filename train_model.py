import os
from ml_pipeline.data import build_clean_dataset
from ml_pipeline.train import train_and_save


def main() -> None:

    cleaned_csv = "cleaned_fake_news.csv"
    if not os.path.exists(cleaned_csv):
        build_clean_dataset("True.csv", "Fake.csv", cleaned_csv)

    metrics = train_and_save(cleaned_csv)
    print({"accuracy": round(metrics["accuracy"], 4)})


if __name__ == "__main__":
    main()


