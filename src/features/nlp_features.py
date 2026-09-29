"""NLP feature extraction: turn each trading day's news headlines into numbers.

OWNER: Person B  (see TEAM_TASKS.md for step-by-step instructions)

Input: a DataFrame from src.data.load (one row per trading day). The relevant
column is `titles`: every headline aligned to that day, joined with " || "
(empty string on days with no news).

Output: a numeric DataFrame with one row per input row (same order / index).

Rules from the assignment:
  * Allowed: lexicon-based sentiment, classical vectorisation (TF-IDF, BoW).
  * NOT allowed: pre-trained embeddings or transformer models (FinBERT, IndoBERT,
    sentence-transformers, LLM embeddings, ...).
  * No leakage: anything that is *learned* from text (TF-IDF vocabulary + IDF
    weights, SVD components) must be fitted on the TRAIN split only. That is why
    this is a class with fit() / transform(): the training script calls
    fit(train) once, then transform(train / val / test).

Status:
  [done] basic_news_features  -- volume + GDELT tone (works now, used as placeholder)
  [TODO] preprocess_title      -- Person B
  [TODO] lexicon features      -- Person B, then set NLP_CONFIG["use_lexicon"] = True
  [TODO] TF-IDF + SVD features -- Person B, then set NLP_CONFIG["use_tfidf"] = True
"""
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import LEXICON_DIR, NLP_CONFIG, SEED

INDONESIAN_STOPWORDS = {
    "yang", "di", "dan", "ini", "itu", "untuk", "pada", "adalah",
    "dengan", "akan", "juga", "oleh", "dalam", "bisa", "karena", "atau",
}

TITLE_SEP = " || "


def split_titles(titles_cell):
    """'title A || title B' -> ['title A', 'title B'] ('' -> [])."""
    if not isinstance(titles_cell, str) or not titles_cell:
        return []
    return [t for t in titles_cell.split(TITLE_SEP) if t.strip()]


def basic_news_features(df):
    """News volume and GDELT's own document tone. Model-free, always on."""
    n = df["n_articles"].astype(float)
    feats = pd.DataFrame(index=df.index)
    feats["has_news"] = (n > 0).astype(float)
    feats["log_n_articles"] = np.log1p(n)
    feats["gdelt_tone_mean"] = df["gdelt_tone_mean"].fillna(0.0)   # 0 = neutral when no news
    feats["share_weekend_news"] = (df["n_non_trading_day_articles"] / n.replace(0, np.nan)).fillna(0.0)
    return feats


# ---------------------------------------------------------------------------
# TODO (Person B): text preprocessing
# ---------------------------------------------------------------------------
def preprocess_title(title):
    """Clean ONE headline and return a list of tokens.

    Suggested steps (justify your choices in the report):
      1. lowercase
      2. remove punctuation / digits (hint: re.sub(r"[^a-z\s]", " ", text))
      3. split on whitespace
      4. (optional) drop Indonesian stopwords, e.g. "yang", "di", "dan", "ke", "dari"

    Example: "Rupiah Melemah 0,5% ke Rp16.200!" -> ["rupiah", "melemah", "ke", "rp"]
    """
    if not isinstance(title, str) or not title.strip():
        return []
    cleaned = re.sub(r"[^a-z\s]", " ", title.lower())
    tokens = cleaned.split()
    return [t for t in tokens if len(t) >= 2 and t not in INDONESIAN_STOPWORDS]


class NLPFeatureExtractor:
    """fit() on the training split, then transform() any split."""

    def __init__(self, config=None):
        self.config = dict(NLP_CONFIG if config is None else config)
        self.lexicon = None      # dict word -> weight, set in _load_lexicon
        self.vectorizer = None   # fitted TfidfVectorizer
        self.svd = None          # fitted TruncatedSVD

    def fit(self, df):
        if self.config["use_lexicon"]:
            self.lexicon = self._load_lexicon()
        if self.config["use_tfidf"]:
            self._fit_tfidf(df)
        return self

    def transform(self, df):
        parts = [basic_news_features(df)]
        if self.config["use_lexicon"]:
            parts.append(self._lexicon_features(df))
        if self.config["use_tfidf"]:
            parts.append(self._tfidf_features(df))
        return pd.concat(parts, axis=1)

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    # -----------------------------------------------------------------------
    # Person B: lexicon-based sentiment (InSet)
    # -----------------------------------------------------------------------
    def _load_lexicon(self):
        """Read the InSet lexicon into a dict {word: weight}.

        Files (download with the command in TEAM_TASKS.md):
            data/lexicon/positive.tsv, data/lexicon/negative.tsv
        Each has a header 'word<TAB>weight'; weights range -5..+5.
        Hint: pd.read_csv(LEXICON_DIR / "positive.tsv", sep="\t")
        Note: a few entries are multi-word phrases -- keeping only single words
        is fine (say so in the report).
        """
        weights = {}
        for filename in ["positive.tsv", "negative.tsv"]:
            path = LEXICON_DIR / filename
            if not path.exists():
                raise FileNotFoundError(f"Lexicon file missing: {path}. Run curl commands in TEAM_TASKS.md.")
            df = pd.read_csv(path, sep="\t")
            for _, row in df.iterrows():
                w = str(row["word"]).strip().lower()
                if " " in w or "(" in w:
                    continue
                try:
                    wt = float(row["weight"])
                    weights.setdefault(w, []).append(wt)
                except (ValueError, TypeError):
                    continue
        return {w: float(np.mean(wts)) for w, wts in weights.items()}

    def _lexicon_features(self, df):
        """One row per day. Suggested columns:
            lex_score_mean  -- mean over that day's headlines of
                               (sum of word weights in the headline)
            lex_pos_share   -- share of headlines with score > 0
            lex_neg_share   -- share of headlines with score < 0
        Days with no headlines -> 0 for every column.
        Must return a DataFrame with index=df.index.
        """
        if self.lexicon is None:
            self.lexicon = self._load_lexicon()

        rows = []
        for titles_cell in df["titles"]:
            titles = split_titles(titles_cell)
            if not titles:
                rows.append((0.0, 0.0, 0.0))
                continue
            scores = []
            for t in titles:
                toks = preprocess_title(t)
                score = sum(self.lexicon.get(tok, 0.0) for tok in toks)
                scores.append(score)
            mean_score = float(np.mean(scores)) if scores else 0.0
            pos_share = float(np.mean([1.0 if s > 0 else 0.0 for s in scores])) if scores else 0.0
            neg_share = float(np.mean([1.0 if s < 0 else 0.0 for s in scores])) if scores else 0.0
            rows.append((mean_score, pos_share, neg_share))

        return pd.DataFrame(
            rows,
            columns=["lex_score_mean", "lex_pos_share", "lex_neg_share"],
            index=df.index,
        )

    # -----------------------------------------------------------------------
    # Person B: TF-IDF + dimensionality reduction
    # -----------------------------------------------------------------------
    def _fit_tfidf(self, df):
        """Fit on the TRAIN split only (this method only ever receives train).

        One document per trading day = all that day's headlines joined.
        """
        docs = df["titles"].fillna("").tolist()
        self.vectorizer = TfidfVectorizer(
            tokenizer=preprocess_title,
            lowercase=False,
            token_pattern=None,
            max_features=self.config["tfidf_max_features"],
            min_df=self.config["tfidf_min_df"],
        )
        X = self.vectorizer.fit_transform(docs)
        self.svd = TruncatedSVD(
            n_components=self.config["svd_components"],
            random_state=SEED,
        ).fit(X)
        return self

    def _tfidf_features(self, df):
        """transform() with the already-fitted vectorizer + SVD (never refit here).
        Return a DataFrame with columns tfidf_svd_0 .. tfidf_svd_{k-1}, index=df.index.
        """
        if self.vectorizer is None or self.svd is None:
            raise RuntimeError("TF-IDF vectorizer and SVD must be fitted before transforming.")
        docs = df["titles"].fillna("").tolist()
        X = self.vectorizer.transform(docs)
        X_svd = self.svd.transform(X)
        cols = [f"tfidf_svd_{i}" for i in range(self.config["svd_components"])]
        return pd.DataFrame(X_svd, columns=cols, index=df.index)
