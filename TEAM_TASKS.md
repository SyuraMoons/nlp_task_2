# Task 2: who does what

| Person | Part | Files | Status |
|---|---|---|---|
| **A** (Harfi) | Repo setup, data split, evaluation, baseline + combined model pipeline, README | `data/`, `src/config.py`, `src/data/`, `src/models/`, `src/evaluation/`, `src/train.py` | ✅ done |
| **B** | NLP feature extraction: preprocessing, lexicon sentiment, TF-IDF | `src/features/nlp_features.py` | ⬜ TODO |
| **C** | EDA + results notebooks, PDF report with pipeline diagram | `notebook/`, `report/` | ⬜ TODO |

The pipeline already runs end to end. B's and C's parts plug into it, so nobody is blocked by anyone else. The one ordering rule: C should re-run the results notebook **after** B finishes.

## Setup (everyone)

```bash
git clone https://github.com/SyuraMoons/nlp_task_2.git
cd nlp_task_2
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train          # should print a results table in ~1 minute
```

Always run commands from the repo root. Work on your own branch (`git checkout -b feat/nlp-features` or `feat/report`) and open a pull request when you're done.

---

## Person B: NLP features

Everything happens in **one file**: `src/features/nlp_features.py`. Each function you need to write has a `TODO` and a docstring that says exactly what to return.

1. **Download the InSet lexicon** (an Indonesian sentiment word list with weights from −5 to +5):
   ```bash
   mkdir -p data/lexicon
   curl -L -o data/lexicon/positive.tsv https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv
   curl -L -o data/lexicon/negative.tsv https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv
   ```
2. **`preprocess_title(title)`**: lowercase, remove punctuation and digits, split into words, drop a small list of Indonesian stopwords.
3. **`_load_lexicon`** and **`_lexicon_features`**: for each headline, add up the weights of its words; for each day, take the mean score plus the share of positive and negative headlines.
4. In `src/config.py`, set `"use_lexicon": True`. Then run:
   ```bash
   python -m src.features.check_nlp_features   # checks shape, NaNs, and that columns match across splits
   python -m src.train                          # new results
   ```
5. **`_fit_tfidf`** and **`_tfidf_features`**: TF-IDF on each day's headlines, then TruncatedSVD down to 20 columns. The code hints are in the docstrings. **Fit only in `_fit_tfidf`**, which only ever receives the train split, and never refit inside `_tfidf_features`.
6. Set `"use_tfidf": True`, then run the check and `python -m src.train` again.
7. **Write notes for C's report** (5–10 bullet points):
   - which stopwords you used
   - why InSet and not VADER (VADER is English-only)
   - why headlines and not bodies (bodies only exist for 2024)
   - why SVD (5,000 words would overfit ~800 training days)
   - which of the two feature types helped more in the results table

Rules: **no pre-trained embeddings or transformers** (no FinBERT, IndoBERT, sentence-transformers, or LLM embeddings). Don't commit the lexicon files; `data/lexicon/` is gitignored.

*Optional extra:* compare lexicon features with only the lexicon turned on vs. lexicon + TF-IDF, and report both rows. That answers the task's hint about which text representation gives the most signal.

---

## Person C: notebooks + report

1. **`notebook/01_eda.ipynb`**: the first cell already loads the data. Fill in the 6 `TODO` sections (each is one plot or table) and write 2–3 sentences under each. These sentences are reused in the report.
2. **`notebook/02_results.ipynb`**: run it after `python -m src.train`, fill in the 4 `TODO`s, and re-run it once B's features are turned on.
3. **Report PDF**: follow `report/REPORT_OUTLINE.md`. It already lists every section, where each fact comes from, and a draft pipeline diagram (Mermaid) that you can render at https://mermaid.live and redraw nicely. Use B's notes for the NLP justification section.
4. Put the final PDF in `report/` as `Task2_Report.pdf`.

---

## Before submitting (everyone)

- [ ] `python -m src.train` runs without errors on a fresh clone
- [ ] `results/` contains the final numbers (with B's features on)
- [ ] both notebooks are run with outputs saved
- [ ] the report PDF is in `report/`
- [ ] **each member** submits on eLOK individually
