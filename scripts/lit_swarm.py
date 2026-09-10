#!/usr/bin/env python3
"""Literature swarm harness for qicert.

Fans out across research lanes x queries. Primary backend: arXiv Atom API
(reliable, unauthenticated, full abstracts). Optional Semantic Scholar Graph
API enrichment for citation counts (degrades gracefully on 429).

Produces deduped, scored shortlists per lane for reading passes.

Usage:
    python scripts/lit_swarm.py --lanes all --limit 25
    python scripts/lit_swarm.py --lanes L1,L4 --limit 25

Outputs (under lit-swarm/):
    raw/arxiv__<qslug>.json      raw arXiv results per query (cache)
    papers/<lane>.json           deduped + scored shortlist per lane
    state.json                   coverage ledger
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "lit-swarm"
RAW = OUT / "raw"
PAPERS = OUT / "papers"
STATE = OUT / "state.json"

ARXIV_SEARCH = "http://export.arxiv.org/api/query"
S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
NS = {"a": "http://www.w3.org/2005/Atom"}

# ---------------------------------------------------------------- lanes ----
LANES = {
    "L1": {
        "name": "TT/QTT compression of transformers + residual/hybrid quantization",
        "queries": [
            'all:"tensor train" AND all:compression AND all:transformer',
            'all:"tensor network" AND all:"language model"',
            'all:"low-rank" AND all:residual AND all:quantization',
            'all:"tensor train" AND all:"large language model"',
            'all:tt AND all:svd AND all:llm',
        ],
    },
    "L2": {
        "name": "Certificates: Lipschitz, gauge optimization, canonical forms",
        "queries": [
            'all:"Lipschitz constant" AND all:certification AND all:neural',
            'all:"matrix product state" AND all:gauge AND all:canonical',
            'all:"operator norm" AND all:"tensor train"',
            'all:certified AND all:robustness AND all:Lipschitz',
            'all:"tensor train" AND all:norm AND all:minimization',
        ],
    },
    "L3": {
        "name": "VLA compression + deployment + latency on edge",
        "queries": [
            'all:"vision-language-action" AND all:compression',
            'all:"vision-language-action" AND all:quantization',
            'all:"vision language action" AND all:edge',
            'all:OpenVLA AND all:quantization',
            'all:VLA AND all:robot AND all:efficiency',
        ],
    },
    "L4": {
        "name": "Formal safety: STL repair, conformal risk control, runtime assurance",
        "queries": [
            'all:"signal temporal logic" AND all:repair',
            'all:"conformal prediction" AND all:robotics AND all:safety',
            'all:"runtime assurance" AND all:learning',
            'all:Lyapunov AND all:"neural network" AND all:certificate',
            'all:"conformal risk control"',
        ],
    },
    "L5": {
        "name": "Calibration/activation-aware compression + repair training",
        "queries": [
            'all:"activation" AND all:aware AND all:compression',
            'all:"post-training" AND all:quantization AND all:recovery',
            'all:"low rank" AND all:calibration AND all:svd',
            'all:LoRA AND all:quantized AND all:model',
            'all:"weight compression" AND all:"fine-tuning" AND all:llm',
        ],
    },
    "L6": {
        "name": "Quantum amplitude estimation + rare events + tensor networks (QM watch)",
        "queries": [
            'all:"amplitude estimation" AND all:safety',
            'all:"amplitude estimation" AND all:"rare event"',
            'all:"quantum inspired" AND all:certification',
            'all:"tensor network" AND all:"quantum machine learning"',
            'all:quantum AND all:"reinforcement learning" AND all:sample',
        ],
    },
}

RELEVANCE_WORDS = [
    "tensor train", "tensor network", "qtt", "low-rank", "low rank",
    "quantiz", "compress", "lipschitz", "certificat", "safety",
    "vision-language-action", "vla", "robot", "libero", "temporal logic",
    "conformal", "lyapunov", "gauge", "calibration", "residual", "amplitude",
    "quantum", "pruning", "attention", "transformer", "adapter", "lora",
]
NEGATIVE_WORDS = [
    "medical imaging", "protein", "weather", "bandgap", "molecular dynamics",
    "chemistry", "crystal",
]


def _score(title: str, abstract: str, year: int | None, query: str) -> float:
    text = (title + " " + abstract).lower()
    if not text.strip():
        return 0.0
    s = 0.0
    s += 3.0 * sum(1 for w in ("tensor train", "tensor network", "qtt",
                               "vision-language-action") if w in text)
    s += 1.0 * sum(1 for w in RELEVANCE_WORDS if w in text)
    s -= 4.0 * sum(1 for w in NEGATIVE_WORDS if w in text)
    if (year or 0) >= 2025:
        s += 2.0
    elif (year or 0) >= 2023:
        s += 1.0
    qterms = [t for t in re.split(r'\W+', query.lower()) if len(t) > 3]
    s += 0.4 * sum(1 for t in qterms if t in text)
    return s


def _fetch(url: str, retries: int = 3) -> bytes | None:
    delay = 3.0
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "qicert-lit-swarm/1.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(delay)
            delay *= 2
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(delay)
            delay *= 2
    return None


def arxiv_search(query: str, limit: int) -> list[dict]:
    """arXiv Atom API search. Returns list of paper dicts."""
    slug = re.sub(r"[^a-z0-9]+", "_", query.lower())[:70]
    cache = RAW / f"arxiv__{slug}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": min(limit, 40),
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    raw = _fetch(f"{ARXIV_SEARCH}?{params}")
    papers: list[dict] = []
    if raw:
        try:
            root = ET.fromstring(raw)
            for entry in root.findall("a:entry", NS):
                axid = (entry.findtext("a:id", "", NS) or "").rsplit("/", 1)[-1]
                title = re.sub(r"\s+", " ", entry.findtext("a:title", "", NS)).strip()
                abstract = re.sub(r"\s+", " ", entry.findtext("a:summary", "", NS)).strip()
                published = entry.findtext("a:published", "", NS)
                year = int(published[:4]) if published[:4].isdigit() else None
                authors = [re.sub(r"\s+", " ", (a.findtext("a:name", "", NS) or ""))
                           for a in entry.findall("a:author", NS)]
                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "ids": {"ArXiv": axid},
                    "url": f"https://arxiv.org/abs/{axid}",
                    "authors": authors[:4],
                    "venue": "arXiv",
                })
        except ET.ParseError as e:
            print(f"  [warn] parse fail for {query}: {e}", file=sys.stderr)
    else:
        print(f"  [warn] arXiv unreachable for: {query}", file=sys.stderr)
    cache.write_text(json.dumps(papers))
    time.sleep(3.0)  # arXiv politeness (1 req / 3 s)
    return papers


def s2_citation_enrich(papers: list[dict]) -> None:
    """Optional: fill citationCount via S2 title match. Silently skips on 429."""
    for p in papers[:15]:  # only top of shortlist
        if p.get("citations") is not None:
            continue
        params = urllib.parse.urlencode({
            "query": p["title"][:120], "limit": 1, "fields": "citationCount,externalIds"})
        raw = _fetch(f"{S2_SEARCH}?{params}", retries=1)
        if not raw:
            return  # S2 unavailable -> stop enrichment, keep going
        try:
            data = json.loads(raw)
            rows = data.get("data") or []
            if rows:
                p["citations"] = rows[0].get("citationCount")
                s2ids = (rows[0].get("externalIds") or {})
                if "DOI" in s2ids:
                    p["ids"]["DOI"] = s2ids["DOI"]
        except (json.JSONDecodeError, KeyError):
            pass
        time.sleep(4.0)


def run_lane(lane_id: str, lane: dict, limit: int, enrich: bool) -> dict:
    seen: dict[str, dict] = {}
    counts = []
    for query in lane["queries"]:
        papers = arxiv_search(query, limit)
        counts.append(len(papers))
        for p in papers:
            key = (p.get("ids") or {}).get("ArXiv") or p.get("title", "").lower()
            if not key:
                continue
            if key in seen:
                seen[key]["query_hits"] += 1
                continue
            p = dict(p)
            p["query_hits"] = 1
            p["lane"] = lane_id
            p["_query"] = query
            p["score"] = round(_score(p["title"], p["abstract"], p["year"], query), 1)
            seen[key] = p
    pool = list(seen.values())
    for p in pool:
        p["score"] = max(p["score"],
                         round(p["score"] + 1.0 * (p["query_hits"] - 1), 1))
    pool.sort(key=lambda p: (-p["query_hits"], -p["score"]))
    shortlist = pool[:limit]
    if enrich:
        s2_citation_enrich(shortlist)
    out = PAPERS / f"{lane_id}.json"
    out.write_text(json.dumps({
        "lane": lane_id, "name": lane["name"], "queries": lane["queries"],
        "query_counts": counts, "unique_found": len(pool),
        "shortlist": [{k: v for k, v in p.items() if k != "_query"}
                      for p in shortlist],
    }, indent=1))
    return {"lane": lane_id, "hits": counts, "unique": len(pool),
            "shortlisted": len(shortlist),
            "with_abstract": sum(1 for p in shortlist if p.get("abstract"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", default="all")
    ap.add_argument("--limit", type=int, default=25, help="shortlist size per lane")
    ap.add_argument("--results-per-query", type=int, default=25)
    ap.add_argument("--enrich", action="store_true",
                    help="attempt S2 citation enrichment (best effort)")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    PAPERS.mkdir(parents=True, exist_ok=True)

    lane_ids = list(LANES) if args.lanes == "all" else args.lanes.split(",")
    state = json.loads(STATE.read_text()) if STATE.exists() else {"runs": []}
    for lid in lane_ids:
        if lid not in LANES:
            print(f"unknown lane {lid}", file=sys.stderr)
            continue
        res = run_lane(lid, LANES[lid], args.results_per_query, args.enrich)
        print(f"{lid}: hits={res['hits']} unique={res['unique']} "
              f"shortlisted={res['shortlisted']} with_abstract={res['with_abstract']}")
        state["runs"].append({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **res})
        STATE.write_text(json.dumps(state, indent=1))
    print(f"\nShortlists in {PAPERS}")


if __name__ == "__main__":
    main()
