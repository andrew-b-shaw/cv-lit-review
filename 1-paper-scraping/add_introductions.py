"""Add each paper's Introduction section to cv_papers_2021_2026.json.

Introductions are only in the PDFs, so each PDF is downloaded into memory,
its Introduction extracted with PyMuPDF, and the PDF discarded (all PDFs
together are ~100 GB). Extracted text is appended to introductions.jsonl as
it goes, so an interrupted run resumes where it stopped; the JSON dataset is
only rewritten once every paper has been attempted.

Each paper gains:
  pdf_url             link to the paper PDF
  introduction        text from "1. Introduction" up to the next section heading,
                      or None if it couldn't be found
  introduction_status "ok", "no-end" (no following heading found; text runs to
                      the end of the first pages), "no-start" (no Introduction
                      heading found), "no-text" (the PDF has no usable text
                      layer; would need OCR) or "error: ..." (download/parse failure)

Usage:
    python add_introductions.py             # all papers
    python add_introductions.py --limit 20  # first 20, for testing (dataset not rewritten)
    python add_introductions.py --redo no-end no-start  # re-extract papers with these statuses
"""

import argparse
import json
import re
import sys
import threading
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
DATASET = HERE / "cv_papers_2021_2026.json"
PROGRESS = HERE / "introductions.jsonl"
WORKERS = 8
MAX_PAGES = 5  # the Introduction always starts on page 1 and rarely runs past page 3
UA = "Mozilla/5.0 (research scraper; titles+abstracts)"
CVF = "https://openaccess.thecvf.com"

# Heading patterns match spaces/tabs rather than \s (which includes newlines), except
# where a line break is explicitly allowed: ECCV's layout puts the section number on its
# own line ("1\nIntroduction", "2\nRelated Work").
# "1. Introduction", "1 . Introduction", "1\nIntroduction", "I. INTRODUCTION", "1. Introduction
# and Related Work"; trailing text is allowed because some headings are longer or have stray text.
START = re.compile(r"^[ \t]*(?:1|I)[ \t]*\.?[ \t]*(?:\n[ \t]*)?Introduction\b[^\n]{0,40}$", re.I | re.M)
# Fallback when section 1 has another name ("1. Motivation", "1. The 3D Revolution").
SECTION_1 = re.compile(r"^[ \t]*1\.?[ \t]+([A-Z][^\n.]{1,60})$", re.M)
# Candidate next-section headings: "2. Related Work", "II. Background", "3. Method", and the
# two-line form "2\nRelated Work" (group 3 is set for the two-line form).
NUMBERED = re.compile(
    r"^[ \t]*(2|II|3|III)[ \t]*\.?(?:[ \t]+|[ \t]*(\n)[ \t]*)([A-Za-z][^\n.]{1,80})$", re.M
)
# Unnumbered Related Work/Background headings, used by some papers.
UNNUMBERED = re.compile(r"^[ \t]*(Related Works?|Background|Prior Work|Preliminaries)[ \t]*$", re.I | re.M)
SECTION_WORDS = re.compile(r"^(related|background|prior|preliminar|method|approach|problem|overview|motivation|literature)", re.I)
# Lines holding only a number: page numbers and the margin line numbers in some PDFs.
NUMBER_LINE = re.compile(r"^[ \t]*\d{1,4}[ \t]*$", re.M)
MIN_TEXT_CHARS = 500  # less than this in the first pages means no usable text layer (needs OCR)


def _is_heading(title, next_line=""):
    """Tell a section title from a numbered list item such as "2. We propose a new loss ...".

    next_line is the line after the candidate: a heading is followed by a new paragraph
    (capitalized) or subsection ("2.1. ..."), while a wrapped list item continues in lowercase.
    """
    words = title.split()
    if SECTION_WORDS.match(title):
        return True
    # First word capitalized, or containing capitals ("vMF-based Similarity ...").
    if not any(c.isupper() for c in words[0]) or words[0] in ("We", "Our"):
        return False
    if len(words) <= 5:
        return True
    long_words = [w for w in words if len(w) >= 4]
    title_case = sum(w[0].isupper() for w in long_words) >= 0.6 * len(long_words)
    new_paragraph = bool(re.match(r"\s*([A-Z]|\d+\.\d)", next_line))
    return len(words) <= 12 and (title_case or new_paragraph)


def _intro_end(rest):
    """Position of the heading that ends the Introduction, or None.

    Takes the earliest section 2 heading or unnumbered Related Work/Background heading.
    Section 3 headings are only used when no section 2 heading is found, since numbered
    contribution lists ("3. Experiments on ...") can look like headings.
    """
    candidates = {"2": [], "3": []}
    for m in NUMBERED.finditer(rest):
        number, two_line, title = m.groups()
        next_line = rest[m.end():].lstrip("\n").split("\n", 1)[0]
        # With the number on its own line, chart labels ("2" / "IN" / "COCO") look like
        # headings too; a real heading is followed by a full line of body text or a
        # subsection ("2.1"). (Not applied to one-line headings: their next line is
        # often a short subheading.)
        if two_line and not (len(next_line.split()) >= 5 or re.match(r"\s*\d+\.\d", next_line)):
            continue
        if _is_heading(title, next_line):
            candidates["2" if number in ("2", "II") else "3"].append(m.start())
    ends = candidates["2"] + [m.start() for m in UNNUMBERED.finditer(rest)]
    ends = ends or candidates["3"]
    return min(ends) if ends else None


def fetch(url, retries=4, timeout=300):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def pdf_url(paper):
    url = paper["url"]
    if url.startswith(CVF):
        # .../content/CVPR2021/html/X_CVPR_2021_paper.html -> .../papers/X_CVPR_2021_paper.pdf
        return url.replace("/html/", "/papers/").removesuffix(".html") + ".pdf"
    # ECCV: the PDF link is only on the poster page.
    page = fetch(url).decode("utf-8", errors="replace")
    links = [l for l in re.findall(r'href="([^"]+\.pdf)"', page) if "supplement" not in l.lower()]
    if not links:
        raise ValueError("no PDF link on paper page")
    return links[0]


def extract_introduction(pdf_bytes):
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        text = "\n".join(doc[i].get_text("text") for i in range(min(MAX_PAGES, len(doc))))
    text = unicodedata.normalize("NFKC", text)  # ligatures like "ﬁ" -> "fi"
    if len(text.strip()) < MIN_TEXT_CHARS:
        return None, "no-text"
    m = START.search(text) or next((s for s in SECTION_1.finditer(text) if _is_heading(s.group(1))), None)
    if not m:
        return None, "no-start"
    rest = text[m.end():]
    end = _intro_end(rest)
    body = rest[:end] if end is not None else rest
    body = NUMBER_LINE.sub("", body)
    body = re.sub(r"-\n(?=[a-z])", "", body)  # re-join words hyphenated across lines
    return re.sub(r"\s+", " ", body).strip(), ("ok" if end is not None else "no-end")


def process(paper):
    rec = {"url": paper["url"], "pdf_url": None, "introduction": None}
    try:
        rec["pdf_url"] = pdf_url(paper)
        rec["introduction"], rec["introduction_status"] = extract_introduction(fetch(rec["pdf_url"]))
    except Exception as e:
        rec["introduction_status"] = f"error: {type(e).__name__}: {e}"
    return rec


def load_progress():
    """Finished papers from earlier runs; errors are dropped so they get retried.

    A paper can appear more than once (e.g. after --redo); the latest line wins.
    """
    done = {}
    if PROGRESS.exists():
        with open(PROGRESS, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if not rec["introduction_status"].startswith("error"):
                    done[rec["url"]] = rec
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only process the first N papers (dataset not rewritten)")
    parser.add_argument("--redo", nargs="+", metavar="STATUS", default=[],
                        help="re-process papers whose saved status is one of these, e.g. --redo no-end no-start")
    parser.add_argument("--redo-under", type=int, metavar="WORDS",
                        help="also re-process papers whose saved introduction is shorter than this")
    args = parser.parse_args()

    papers = json.loads(DATASET.read_text(encoding="utf-8"))
    todo_papers = papers[: args.limit] if args.limit else papers
    done = load_progress()
    done = {
        url: rec for url, rec in done.items()
        if rec["introduction_status"] not in args.redo
        and not (args.redo_under and rec["introduction"] and len(rec["introduction"].split()) < args.redo_under)
    }
    todo = [p for p in todo_papers if p["url"] not in done]
    print(f"{len(todo_papers)} papers, {len(todo_papers) - len(todo)} already done, {len(todo)} to process")

    lock = threading.Lock()
    start, count = time.time(), 0
    with open(PROGRESS, "a", encoding="utf-8") as out, ThreadPoolExecutor(WORKERS) as pool:
        for rec in pool.map(process, todo):
            with lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                if not rec["introduction_status"].startswith("error"):
                    done[rec["url"]] = rec
                count += 1
                if rec["introduction_status"].startswith("error"):
                    print(f"  {rec['url']}: {rec['introduction_status']}", file=sys.stderr)
                if count % 100 == 0 or count == len(todo):
                    rate = count / (time.time() - start)
                    print(f"  {count}/{len(todo)} ({rate * 60:.0f}/min, ~{(len(todo) - count) / rate / 60:.0f} min left)", flush=True)

    statuses = {}
    for p in todo_papers:
        s = done[p["url"]]["introduction_status"] if p["url"] in done else "error"
        statuses[s] = statuses.get(s, 0) + 1
    print("Status counts:", statuses)

    if args.limit:
        print("--limit given: dataset not rewritten.")
        return
    failed = 0
    for p in papers:
        rec = done.get(p["url"])
        if rec is None:  # still failing after retries; rerunning the script tries again
            failed += 1
            rec = {"pdf_url": None, "introduction": None, "introduction_status": "error"}
        p["pdf_url"] = rec["pdf_url"]
        p["introduction"] = rec["introduction"]
        p["introduction_status"] = rec["introduction_status"]
    DATASET.write_text(json.dumps(papers, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {DATASET} ({failed} papers failed to download; rerun to retry them)")


if __name__ == "__main__":
    main()
