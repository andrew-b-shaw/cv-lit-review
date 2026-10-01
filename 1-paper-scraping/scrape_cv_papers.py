"""Scrape titles and abstracts for CVPR 2021, ICCV 2021, CVPR 2026 and ECCV 2026.

ICCV is held in odd years and ECCV in even years, so the "ICCV/ECCV" slot is
ICCV for 2021 and ECCV for 2026.

Sources:
  - CVPR/ICCV: CVF Open Access (openaccess.thecvf.com), one page per paper.
  - ECCV 2026: ECVA virtual site (eccv.ecva.net); paper list from its JSON
    feed, abstracts from each poster page.

Standard library only. Fetched pages are cached in ./cache so reruns resume.
"""

import hashlib
import html
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = HERE / "cv_papers_2021_2026.json"
WORKERS = 12
UA = "Mozilla/5.0 (research scraper; titles+abstracts)"

CVF = "https://openaccess.thecvf.com"
ECVA = "https://eccv.ecva.net"


def fetch(url, retries=4):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / (hashlib.sha1(url.encode()).hexdigest() + ".html")
    if path.exists():
        return path.read_text(encoding="utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                text = r.read().decode("utf-8", errors="replace")
            path.write_text(text, encoding="utf-8")
            return text
        except Exception as e:
            if attempt == retries - 1:
                print(f"  FAILED {url}: {e}", file=sys.stderr)
                return None
            time.sleep(2 ** attempt)


def clean(s):
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def scrape_cvf(conf, year):
    key = f"{conf}{year}"
    index = fetch(f"{CVF}/{key}?day=all")
    links = re.findall(
        r'<dt class="ptitle">.*?<a href="(/content/[^"]+\.html)">(.*?)</a>', index, re.S
    )
    print(f"{conf} {year}: {len(links)} papers")

    def one(item):
        href, title = item
        page = fetch(CVF + href)
        m = re.search(r'<div id="abstract">(.*?)</div>', page or "", re.S)
        return {
            "conference": conf,
            "year": year,
            "title": clean(title),
            "abstract": clean(m.group(1)) if m else None,
            "url": CVF + href,
        }

    return run(one, links)


def scrape_eccv(year):
    feed = json.loads(fetch(f"{ECVA}/static/virtual/data/eccv-{year}-orals-posters.json"))
    # Orals/spotlights also appear as posters; keep one entry per title.
    seen, events = set(), []
    for ev in sorted(feed["results"], key=lambda e: e["eventtype"] != "Poster"):
        t = clean(ev["name"]).lower()
        if t not in seen:
            seen.add(t)
            events.append(ev)
    print(f"ECCV {year}: {len(events)} papers")

    def one(ev):
        url = ECVA + ev["virtualsite_url"]
        page = fetch(url)
        m = re.search(r'<div class="abstract-text-inner">(.*?)</div>', page or "", re.S)
        return {
            "conference": "ECCV",
            "year": year,
            "title": clean(ev["name"]),
            "abstract": clean(m.group(1)) if m else None,
            "url": url,
        }

    return run(one, events)


def run(fn, items):
    out = []
    with ThreadPoolExecutor(WORKERS) as pool:
        for i, rec in enumerate(pool.map(fn, items), 1):
            out.append(rec)
            if i % 500 == 0:
                print(f"  {i}/{len(items)}")
    return out


def main():
    papers = []
    papers += scrape_cvf("CVPR", 2021)
    papers += scrape_cvf("ICCV", 2021)
    papers += scrape_cvf("CVPR", 2026)
    papers += scrape_eccv(2026)

    OUT.write_text(json.dumps(papers, ensure_ascii=False, indent=1), encoding="utf-8")
    missing = sum(1 for p in papers if not p["abstract"])
    print(f"Wrote {len(papers)} papers to {OUT} ({missing} missing abstracts)")


if __name__ == "__main__":
    main()
