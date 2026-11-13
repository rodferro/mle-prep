"""
Bundle the problems into site/problems.json and write the Anki import file.

    python scripts/build_site.py [--base-url https://rodferro.github.io/mle-prep/]

Reads catalog.json plus problems/, tests/ and solutions/. Output:
    site/problems.json   loaded by the web page
    site/anki.txt        Anki text import (also downloadable from the site)
"""

import argparse
import ast
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "https://rodferro.github.io/mle-prep/"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def docstring(source: str) -> str:
    return (ast.get_docstring(ast.parse(source), clean=True) or "").strip()


def stub_without_docstring(source: str) -> str:
    """The editor starts with the code only; the statement is shown beside it."""
    tree = ast.parse(source)
    first = tree.body[0] if tree.body else None
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant):
        lines = source.splitlines(keepends=True)
        source = "".join(lines[first.end_lineno:])
    return source.lstrip("\n")


def build(base_url: str) -> list:
    catalog = json.loads(read(ROOT / "catalog.json"))
    conftest = read(ROOT / "tests" / "conftest.py")
    out = []
    for item in catalog:
        mod = item["module"]
        problem_src = read(ROOT / "problems" / f"{mod}.py")
        out.append(
            {
                **item,
                "statement": docstring(problem_src),
                "stub": stub_without_docstring(problem_src),
                "solution": read(ROOT / "solutions" / f"{mod}.py"),
                "test_file": f"test_{mod}.py",
                "tests": read(ROOT / "tests" / f"test_{mod}.py"),
                "url": f"{base_url}#{item['id']}",
            }
        )
    return out, conftest


def anki_lines(problems: list) -> str:
    def field(s: str) -> str:
        # Anki text import: tabs/newlines must not appear raw inside a field.
        return s.replace("\t", "    ").replace("\r", "").replace("\n", "<br>")

    rows = [
        "#separator:tab",
        "#html:true",
        "#notetype:Basic",
        "#deck:MLE Prep",
        "#guid column:1",
        "#tags column:4",
    ]
    for p in problems:
        front = (
            f"<b>{html.escape(p['id'].upper())} · {html.escape(p['title'])}</b> "
            f"<i>({html.escape(p['difficulty'])})</i><br><br>"
            f"<pre style='text-align:left;white-space:pre-wrap'>{html.escape(p['statement'])}</pre>"
            f"<a href='{p['url']}'>Solve it in the browser →</a>"
        )
        back = (
            f"<pre style='text-align:left;white-space:pre-wrap;font-size:0.85em'>"
            f"{html.escape(p['solution'])}</pre>"
        )
        tags = " ".join(["mle-prep", *p["tags"]])
        rows.append("\t".join([f"mle-prep-{p['id']}", field(front), field(back), tags]))
    return "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = ap.parse_args()

    problems, conftest = build(args.base_url)
    site = ROOT / "site"
    site.mkdir(exist_ok=True)
    (site / "problems.json").write_text(
        json.dumps({"conftest": conftest, "problems": problems}, indent=1), encoding="utf-8"
    )
    (site / "anki.txt").write_text(anki_lines(problems), encoding="utf-8")
    print(f"built {len(problems)} problems -> site/problems.json, site/anki.txt")


if __name__ == "__main__":
    main()
