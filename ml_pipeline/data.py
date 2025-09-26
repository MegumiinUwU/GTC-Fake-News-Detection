import os
import pandas as pd


def load_raw_data(true_csv_path: str, fake_csv_path: str) -> pd.DataFrame:

    true_df = pd.read_csv(true_csv_path)
    fake_df = pd.read_csv(fake_csv_path)

    true_df["label"] = 1
    fake_df["label"] = 0

    combined = pd.concat([true_df, fake_df], axis=0, ignore_index=True)
    return combined


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    df = df.drop_duplicates().dropna().reset_index(drop=True)
    return df


def build_clean_dataset(true_csv_path: str, fake_csv_path: str, output_csv_path: str | None = None) -> pd.DataFrame:

    df = load_raw_data(true_csv_path, fake_csv_path)
    df = clean_dataframe(df)

    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
        df.to_csv(output_csv_path, index=False)

    return df


