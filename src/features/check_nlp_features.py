"""Quick self-check for Person B's NLP features. Run after each change:

    python -m src.features.check_nlp_features

It fits the extractor on TRAIN only, transforms all three splits, and checks that
the output has the right shape, no missing values, and identical columns.
"""
from src.config import NLP_CONFIG
from src.data.load import load_all
from src.features.nlp_features import NLPFeatureExtractor, preprocess_title


def main():
    data = load_all()
    print("NLP_CONFIG:", NLP_CONFIG)
    try:
        print("preprocess_title example:",
              preprocess_title("Rupiah Melemah 0,5% ke Rp16.200 Jelang Keputusan The Fed!"))
    except NotImplementedError as e:
        print("preprocess_title: not implemented yet --", e)

    extractor = NLPFeatureExtractor().fit(data["train"])
    cols = None
    for name, df in data.items():
        feats = extractor.transform(df)
        assert len(feats) == len(df), f"{name}: expected {len(df)} rows, got {len(feats)}"
        assert (feats.index == df.index).all(), f"{name}: index must equal df.index"
        assert not feats.isna().any().any(), f"{name}: contains NaN -> fill days without news with 0"
        cols = cols or list(feats.columns)
        assert list(feats.columns) == cols, f"{name}: columns differ between splits"
        print(f"{name:5s} OK  shape={feats.shape}")
    print("\nfeatures:", cols)
    print(extractor.transform(data["train"]).describe().T.round(3))


if __name__ == "__main__":
    main()
