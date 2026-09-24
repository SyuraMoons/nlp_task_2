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
import numpy as np
import pandas as pd

from src.config import LEXICON_DIR, NLP_CONFIG, SEED

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
      2. remove punctuation / digits (hint: re.sub(r"[^a-z\\s]", " ", text))
      3. split on whitespace
      4. (optional) drop Indonesian stopwords, e.g. "yang", "di", "dan", "ke", "dari"

    Example: "Rupiah Melemah 0,5% ke Rp16.200!" -> ["rupiah", "melemah", "ke", "rp"]
    """
    raise NotImplementedError("Person B: implement preprocess_title")


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
    # TODO (Person B): lexicon-based sentiment (InSet)
    # -----------------------------------------------------------------------
    def _load_lexicon(self):
        """Read the InSet lexicon into a dict {word: weight}.

        Files (download with the command in TEAM_TASKS.md):
            data/lexicon/positive.tsv, data/lexicon/negative.tsv
        Each has a header 'word<TAB>weight'; weights range -5..+5.
        Hint: pd.read_csv(LEXICON_DIR / "positive.tsv", sep="\\t")
        Note: a few entries are multi-word phrases -- keeping only single words
        is fine (say so in the report).
        """
        raise NotImplementedError("Person B: implement _load_lexicon")

    def _lexicon_features(self, df):
        """One row per day. Suggested columns:
            lex_score_mean  -- mean over that day's headlines of
                               (sum of word weights in the headline)
            lex_pos_share   -- share of headlines with score > 0
            lex_neg_share   -- share of headlines with score < 0
        Days with no headlines -> 0 for every column.
        Must return a DataFrame with index=df.index.
        """
        raise NotImplementedError("Person B: implement _lexicon_features")

    # -----------------------------------------------------------------------
    # TODO (Person B): TF-IDF + dimensionality reduction
    # -----------------------------------------------------------------------
    def _fit_tfidf(self, df):
        """Fit on the TRAIN split only (this method only ever receives train).

        One document per trading day = all that day's headlines joined.
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            self.vectorizer = TfidfVectorizer(
                tokenizer=..., lowercase=False,          # reuse preprocess_title
                max_features=self.config["tfidf_max_features"],
                min_df=self.config["tfidf_min_df"])
            X = self.vectorizer.fit_transform(docs)
            self.svd = TruncatedSVD(self.config["svd_components"], random_state=SEED).fit(X)
        """
        raise NotImplementedError("Person B: implement _fit_tfidf")

    def _tfidf_features(self, df):
        """transform() with the already-fitted vectorizer + SVD (never refit here).
        Return a DataFrame with columns tfidf_svd_0 .. tfidf_svd_{k-1}, index=df.index.
        """
        raise NotImplementedError("Person B: implement _tfidf_features")
