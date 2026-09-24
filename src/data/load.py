"""Load the train / val / test splits written by src/data/make_splits.py."""
import pandas as pd

from src.config import DATA_DIR


def load_split(name):
    """name: 'train', 'val' or 'test'."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    df["titles"] = df["titles"].fillna("")
    return df


def load_all():
    return {name: load_split(name) for name in ("train", "val", "test")}
