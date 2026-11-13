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

## Layout

```
catalog.json        problem list: id, title, difficulty, tags
problems/           stubs (the full problem statement is the module docstring)
tests/              pytest suites, one per problem
solutions/          reference solutions with notes on complexity and follow-ups
site/               the web app (index.html, app.js, worker.js, style.css)
scripts/build_site.py   bundles everything into site/problems.json and site/anki.txt
```

## Practice locally

```bash
pip install pytest
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
[`/anki.txt`](https://rodferro.github.io/mle-prep/anki.txt). In Anki, use **File → Import**
and pick the file. It creates Basic notes in the **MLE Prep** deck:

- **Front:** the problem statement, plus a link that opens it in the browser
- **Back:** the reference solution

Each note has a stable ID (`mle-prep-p03`), so importing again after adding or editing
problems updates the existing cards and keeps your review history.

## Adding a problem

1. Add `problems/pNN_name.py` (the docstring is the statement, followed by the stub),
   `tests/test_pNN_name.py` (load the module with `load("pNN_name")`), and
   `solutions/pNN_name.py`.
2. Add an entry to `catalog.json`.
3. Push. GitHub Actions checks that every reference solution passes, then redeploys the site.
