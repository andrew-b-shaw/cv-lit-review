"""Draw a random sample of papers for a hand audit, independent of any model's predictions.

Picks N papers at random from each conference, so every paper in a conference has
the same chance of being chosen. Because the sample doesn't depend on the category
a model assigned, it supports unbiased estimates of each model's recall and of how
many papers each model missed. Model predictions are left out so they can't influence
the hand labels; join them on url afterwards.

Papers in pre_audit.csv are skipped: they were used to tune the prompt and taxonomy,
so models may do better on them than on unseen papers.

The current random_audit.csv was not drawn by this version of the script; see
random_audit_history.md.

Writes random_audit.csv (with a byte-order mark so Excel opens it with the right
encoding), with blank category_1..category_3 columns to fill in by hand (leave
all blank if the paper has no downstream task).

Usage:
    python sample_random_audit.py                     # 25 per conference
    python sample_random_audit.py --per-conference 50
"""

import argparse
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "1-paper-scraping" / "cv_papers_2021_2026_with_intros.json"
PRE_AUDIT = HERE.parent / "1-paper-scraping" / "pre_audit.csv"
CONFERENCES = ["CVPR 2021", "ICCV 2021", "CVPR 2026", "ECCV 2026"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-conference", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=HERE / "random_audit.csv")
    args = parser.parse_args()

    papers = json.loads(DATASET.read_text(encoding="utf-8"))
    with open(PRE_AUDIT, encoding="utf-8-sig", newline="") as f:
        pre_audit = {r["url"] for r in csv.DictReader(f)}
    rng = random.Random(args.seed)
    rows = []
    for conf in CONFERENCES:
        pool = [p for p in papers if f"{p['conference']} {p['year']}" == conf and p["url"] not in pre_audit]
        for p in rng.sample(pool, args.per_conference):
            rows.append({
                "audit_id": len(rows) + 1,
                "conference": p["conference"],
                "year": p["year"],
                "title": p["title"],
                "url": p["url"],
                "abstract": p["abstract"],
                "introduction": p["introduction"],
                # Filled in by hand:
                "category_1": "",
                "category_2": "",
                "category_3": "",
            })
        print(f"  {conf}: {args.per_conference} of {len(pool)} papers")

    if args.output.exists():
        raise SystemExit(f"{args.output} already exists; it may hold hand labels, so it won't be overwritten")
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.output} ({len(rows)} papers)")


if __name__ == "__main__":
    main()
