"""Test how a set of regex terms lines up with one category's papers.

For each model, prints overall and by year group (2021 = CVPR + ICCV 2021,
2026 = CVPR + ECCV 2026):

  1) matching papers in category: of the papers that match any term, the share the
     model put in the category;
  2) category papers matching:    of the papers the model put in the category, the
     share that match any term;
  and, as a baseline for 2), the share of papers outside the category that match.

A paper matches if any term appears in any of the searched fields (title, abstract
and introduction by default). Terms are Python regular expressions, matched
case-insensitively; use (?-i:VQA) for a case-sensitive term. Plain words match
inside longer words ("robot" also matches "robots" and "robotic"); use \\b for a
word boundary.

The category can be its full taxonomy name or any unique part of it, ignoring case.
Terms can be given directly with --terms, or by name from a term set saved in
regex_terms.json with --term-set, or both.

Usage:
    python regex_audit.py --category caregiving --terms robot embodied
    python regex_audit.py --category "Creative Tasks" --term-set "Creative Tasks" --by-term
    python regex_audit.py --category "obtain and understand" --term-set "Creative Tasks" --models gpt-4o
    python regex_audit.py --category creative --terms "text[\\s-]to[\\s-]image" --fields title abstract
"""

import argparse
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATASET = ROOT / "1-paper-scraping" / "cv_papers_2021_2026_with_intros.json"
CLASSIFICATIONS = ROOT / "3-paper-classification"
TAXONOMY = ROOT / "2-model-comparison" / "task_taxonomy.json"
TERM_SETS = HERE / "regex_terms.json"
# Read directly rather than importing classify_eval, which pulls in scikit-learn/SciPy.
CATEGORIES = [t["task"] for t in json.loads(TAXONOMY.read_text(encoding="utf-8"))]

FIELDS = ["title", "abstract", "introduction"]
YEAR_GROUPS = {"2021": ("CVPR 2021", "ICCV 2021"), "2026": ("CVPR 2026", "ECCV 2026")}
COLUMNS = ("all years", *YEAR_GROUPS)


def resolve_category(name):
    """Full taxonomy name for an exact or unique partial name, ignoring case."""
    exact = [c for c in CATEGORIES if c.lower() == name.lower()]
    partial = [c for c in CATEGORIES if name.lower() in c.lower()]
    found = exact or partial
    if len(found) != 1:
        raise SystemExit(f"--category {name!r} matches {found or 'no category'}; choose from:\n  " + "\n  ".join(CATEGORIES))
    return found[0]


def load_terms(terms, term_sets):
    saved = json.loads(TERM_SETS.read_text(encoding="utf-8"))["categories"] if term_sets else {}
    for name in term_sets:
        if name not in saved:
            raise SystemExit(f"--term-set {name!r} isn't in {TERM_SETS.name}; saved sets: {list(saved)}")
    patterns = list(dict.fromkeys([*terms, *(t for name in term_sets for t in saved[name])]))
    if not patterns:
        raise SystemExit("give at least one term with --terms or --term-set")
    try:
        return {t: re.compile(t, re.IGNORECASE) for t in patterns}
    except re.error as e:
        raise SystemExit(f"bad regex {e.pattern!r}: {e}")


def load_categories(model):
    with open(CLASSIFICATIONS / f"{model}_classifications.csv", encoding="utf-8", newline="") as f:
        return {r["url"]: {r[f"category_{i}"] for i in (1, 2, 3) if r[f"category_{i}"]} for r in csv.DictReader(f)}


def ratio(k, n):
    return f"{k / n:.1%} ({k:,}/{n:,})" if n else "-"


def by_year(papers, fn):
    """fn(subset) for all papers and for each year group."""
    return [fn(papers)] + [fn([p for p in papers if p["group"] in groups]) for groups in YEAR_GROUPS.values()]


def print_row(label, cells, width):
    print(f"  {label:<{width}} " + " ".join(f"{c:>22}" for c in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", required=True, help="taxonomy category, or a unique part of its name")
    parser.add_argument("--terms", nargs="+", default=[], metavar="REGEX", help="terms to search for")
    parser.add_argument("--term-set", nargs="+", default=[], metavar="NAME",
                        help=f"named term set(s) from {TERM_SETS.name}, added to --terms")
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "qwen3.8-flash"])
    parser.add_argument("--fields", nargs="+", default=FIELDS, choices=FIELDS, help="paper text to search")
    parser.add_argument("--by-term", action="store_true", help="also break both shares down by term")
    args = parser.parse_args()

    category = resolve_category(args.category)
    terms = load_terms(args.terms, args.term_set)
    papers = [
        {"url": p["url"], "group": f"{p['conference']} {p['year']}"}
        | {"hits": {t for t, rx in terms.items() if any(rx.search(p.get(f) or "") for f in args.fields)}}
        for p in json.loads(DATASET.read_text(encoding="utf-8"))
    ]

    print(f"Category: {category}")
    print(f"Terms:    {' | '.join(terms)}")
    print(f"Fields:   {', '.join(args.fields)}")
    width = max([33, *(len(t) + 4 for t in terms)] if args.by_term else [33])
    for model in args.models:
        predicted = load_categories(model)
        for p in papers:
            p["in"] = category in predicted[p["url"]]
        print(f"\n{model}")
        print_row("", COLUMNS, width)

        print_row("1) matching papers in category", by_year(papers, lambda ps: ratio(
            sum(p["in"] for p in ps if p["hits"]), sum(bool(p["hits"]) for p in ps))), width)
        if args.by_term:
            for t in terms:
                print_row(f"    {t}", by_year(papers, lambda ps: ratio(
                    sum(p["in"] for p in ps if t in p["hits"]), sum(t in p["hits"] for p in ps))), width)

        print_row("2) category papers matching", by_year(papers, lambda ps: ratio(
            sum(bool(p["hits"]) for p in ps if p["in"]), sum(p["in"] for p in ps))), width)
        if args.by_term:
            for t in terms:
                print_row(f"    {t}", by_year(papers, lambda ps: ratio(
                    sum(t in p["hits"] for p in ps if p["in"]), sum(p["in"] for p in ps))), width)

        print_row("   baseline: other papers matching", by_year(papers, lambda ps: ratio(
            sum(bool(p["hits"]) for p in ps if not p["in"]), sum(not p["in"] for p in ps))), width)


if __name__ == "__main__":
    main()
