# MLE Prep

LeetCode-style practice problems for machine learning engineering interviews. You write a
function or class, and tests tell you whether it's correct. A reference solution is
included for each problem.

**Practice in the browser:** https://rodferro.github.io/mle-prep/
Python and pytest run in the browser through [Pyodide](https://pyodide.org), so there's
nothing to install. Nothing is saved, and each visit starts from a clean stub.

| # | Problem | Difficulty |
|---|---------|------------|
| P01 | Linear regression with gradient descent | Easy |
| P02 | Precision and recall from a confusion matrix | Easy |
| P03 | Near-duplicate agreement events in a stream | Medium |
| P04 | Rolling model performance metrics | Medium |
| P05 | Safely parse and preprocess large text documents | Medium |
| P06 | Sessionization *(pandas)* | Medium |
| P07 | Point-in-time feature join *(pandas)* | Hard |
| P08 | Velocity features for fraud detection *(pandas)* | Medium |
| P09 | Weekly cohort retention table *(pandas)* | Medium |
| P10 | Cleaning a messy export *(pandas)* | Medium |
| P11 | ROC-AUC and average precision from scratch | Medium |
| P12 | Leakage-safe train/validation splits | Medium |
| P13 | Drift detection with PSI | Easy |
| P14 | Choosing a decision threshold under costs and capacity | Medium |
| P15 | Probability calibration: reliability bins, ECE, Platt scaling | Medium |
| P16 | K-nearest neighbors classifier *(numpy)* | Easy |
| P17 | Gaussian Naive Bayes *(numpy)* | Easy |
| P18 | Logistic regression with L2 regularization *(numpy)* | Medium |
| P19 | Principal Component Analysis *(numpy)* | Medium |
| P20 | K-means clustering *(numpy)* | Medium |
| P21 | Best split for a decision tree node | Medium |
| P22 | Linear SVM with hinge loss *(numpy)* | Medium |

Pandas problems require pandas: the tests check that you return DataFrames. P16–P22
are classic algorithms implemented with numpy, in a scikit-learn style API (`fit`,
`predict`, ...), adapted from [MLfromscratch](https://github.com/rodferro/MLfromscratch).
On the site, pandas and numpy load only when you open a problem that needs them (a few
seconds the first time).

## Layout

```
catalog.json        problem list: id, title, difficulty, tags, extra packages
requirements.txt    what you need to run the tests locally
problems/           stubs (the full problem statement is the module docstring)
tests/              pytest suites, one per problem
solutions/          reference solutions with notes on complexity and follow-ups
site/               the web app (index.html, app.js, worker.js, style.css)
scripts/build_site.py   bundles everything into site/problems.json and site/anki.txt
```

## Practice locally

```bash
pip install -r requirements.txt
pytest tests/test_p03_near_duplicates.py   # your code in problems/
USE_SOLUTIONS=1 pytest                     # same tests, reference solutions
```

To run the site locally:

```bash
python scripts/build_site.py
python -m http.server -d site 8000         # open http://localhost:8000
```

## Anki

The site serves an Anki import file at
[`/anki.txt`](https://rodferro.github.io/mle-prep/anki.txt). It targets a note type named
**MLE Prep** with these fields, in order:

| # | Field | Value |
|---|-------|-------|
| 1 | ID | `P03` (title link: `https://rodferro.github.io/mle-prep/#{{ID}}`) |
| 2 | TitleSlug | module name, e.g. `p03_near_duplicates` |
| 3 | Title | problem title |
| 4 | Content | statement in `<pre style="white-space: pre-wrap">` |
| 5 | Difficulty | `Easy` / `Medium` / `Hard` |
| 6 | CodeSnippet | the stub (HTML-escaped; wrap in `<pre>` in the template) |
| 7 | Code | the reference solution (HTML-escaped) |

Tags go into Anki's own tags (`mle-prep` plus the catalog tags). In Anki, use
**File → Import** and pick the file. Notes go into the **MLE Prep** deck.

Each note has a stable GUID (`mle-prep-p03`), so importing again after adding or editing
problems updates the existing notes and keeps your review history.

## Adding a problem

1. Add `problems/pNN_name.py` (the docstring is the statement, followed by the stub),
   `tests/test_pNN_name.py` (load the module with `load("pNN_name")`), and
   `solutions/pNN_name.py`.
2. Add an entry to `catalog.json`.
3. Push. GitHub Actions checks that every reference solution passes, then redeploys the site.
