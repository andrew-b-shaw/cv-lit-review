"""Classify every paper in the dataset with one model.

Uses the same prompt (title, abstract and introduction), reply parsing and model
setup as the evaluation in 2-model-comparison, so results match what was measured
there. Requests run in parallel threads; each result is appended to
<model>_progress.jsonl as it arrives, so an interrupted run resumes where it
stopped (papers that errored are retried). When every paper has a result, the
output CSV is written.

Output: <model>_classifications.csv with one row per paper: conference, year,
title, url, category_1..category_3 (all empty when the model answered "Unclear"),
used_introduction (False when the provider's content filter blocked the
introduction, with an error or repeated empty replies, so only the title and
abstract were used) and raw_reply. Replies with no usable answer (empty, "[]" or
garbled) are retried.

Usage:
    python classify_papers.py                        # Qwen3.8-Flash on all papers
    python classify_papers.py --model llama-3.3-70b
    python classify_papers.py --limit 50             # first 50 papers, for testing
"""

import argparse
import csv
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "2-model-comparison"))

from classifiers import MODELS, _together_client  # noqa: E402
from classify_eval import MAX_CATEGORIES, PROMPT_STYLES, UNCLEAR  # noqa: E402

DATASET = ROOT / "1-paper-scraping" / "cv_papers_2021_2026_with_intros.json"
WORKERS = 16
RETRIES = 3  # per paper, on top of the Together client's own retries (e.g. for dropped streams)
CONTENT_FILTER = "DataInspectionFailed"  # error text when Qwen's provider blocks the input


def classify_paper(clf, paper):
    build_messages, parse = PROMPT_STYLES[clf.prompt_style]
    rec = {"url": paper["url"], "pred": [], "raw_reply": "", "error": "", "used_introduction": True}
    for attempt in range(RETRIES):
        intro = paper["introduction"] if rec["used_introduction"] else ""
        try:
            rec["raw_reply"] = clf.classify(build_messages(paper["title"], paper["abstract"], intro))
            rec["pred"] = parse(rec["raw_reply"])
            if not rec["pred"]:
                # Empty, "[]" or garbled reply: no category and not "Unclear" either.
                raise ValueError(f"no category or {UNCLEAR!r} in reply: {rec['raw_reply'][:100]!r}")
            rec["error"] = ""
            break
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            # The provider's content filter can reject the input (an explicit error) or
            # silently return an empty reply, e.g. for AI-safety papers that quote harmful
            # prompts in their introduction. Fall back to the title and abstract rather
            # than retrying the same input; one empty reply is retried first, since
            # those are also occasionally transient.
            blocked = CONTENT_FILTER in str(e) or (not rec["raw_reply"].strip() and attempt >= 1)
            if blocked and rec["used_introduction"]:
                rec["used_introduction"] = False
                continue
            time.sleep(2 ** attempt)
    return rec


def load_progress(path):
    """Results from earlier runs; errors and unusable replies are dropped so they get retried.

    A paper can appear more than once (after a retry); the latest line wins.
    """
    done = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if not rec["error"] and rec["pred"]:
                    done[rec["url"]] = rec
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen3.8-flash", choices=list(MODELS))
    parser.add_argument("--limit", type=int, help="only classify the first N papers")
    parser.add_argument("--workers", type=int, default=WORKERS, help="parallel requests")
    args = parser.parse_args()

    clf = MODELS[args.model]
    progress_path = HERE / f"{args.model}_progress.jsonl"
    output_path = HERE / f"{args.model}_classifications.csv"

    papers = json.loads(DATASET.read_text(encoding="utf-8"))
    if args.limit:
        papers = papers[: args.limit]
    done = load_progress(progress_path)
    todo = [p for p in papers if p["url"] not in done]
    print(f"{args.model}: {len(papers)} papers, {len(papers) - len(todo)} already done, {len(todo)} to classify")

    _together_client()  # create the shared client before the threads start
    lock = threading.Lock()
    start, count, errors = time.time(), 0, 0
    with open(progress_path, "a", encoding="utf-8") as out, ThreadPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(classify_paper, clf, p) for p in todo]):
            rec = future.result()
            with lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                count += 1
                if rec["error"]:
                    errors += 1
                    print(f"  error on {rec['url']}: {rec['error']}", file=sys.stderr)
                else:
                    done[rec["url"]] = rec
                if count % 250 == 0 or count == len(todo):
                    rate = count / (time.time() - start)
                    print(f"  {count}/{len(todo)} ({rate * 60:.0f}/min, ~{(len(todo) - count) / rate / 60:.0f} min left)", flush=True)

    missing = [p for p in papers if p["url"] not in done]
    if missing:
        print(f"{len(missing)} papers still failing; rerun to retry them before the CSV is written.")
        raise SystemExit(1)

    fields = (["conference", "year", "title", "url"] + [f"category_{i}" for i in range(1, MAX_CATEGORIES + 1)]
              + ["used_introduction", "raw_reply"])
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in papers:
            rec = done[p["url"]]
            cats = [c for c in rec["pred"] if c != UNCLEAR]
            row = {k: p[k] for k in ("conference", "year", "title", "url")}
            row.update({f"category_{i + 1}": cats[i] if i < len(cats) else "" for i in range(MAX_CATEGORIES)})
            row["used_introduction"] = rec.get("used_introduction", True)
            row["raw_reply"] = rec["raw_reply"]
            w.writerow(row)

    counts = Counter(c for p in papers for c in done[p["url"]]["pred"])
    print(f"Wrote {output_path} ({len(papers)} papers)")
    for c, n in counts.most_common():
        print(f"  {c:<52} {n:>6}")


if __name__ == "__main__":
    main()
