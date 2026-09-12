"""Outcome-independent sampling and joint publication assessment."""
from collections import defaultdict

from .controlled_study import CorrespondenceBatch, MATCH_SYSTEM, digest


def matching_sample(manifest, limit=100):
    groups = defaultdict(list)
    for task in manifest["tasks"]:
        groups[task["domain"]].append(task["task_id"])
    for domain in groups:
        groups[domain].sort(key=lambda tid: digest(["matching-sample-v1", domain, tid]))
    selected = []
    # Round-robin yields equal domain allocations when enough tasks are available.
    while len(selected) < limit and any(groups.values()):
        for domain in sorted(groups):
            if groups[domain] and len(selected) < limit:
                selected.append(groups[domain].pop(0))
    return selected


def assess_publications(candidate, papers, caller, identity):
    sources = {p["paper_id"]: p for p in papers}
    if len(sources) != len(papers) or not papers:
        raise ValueError("Joint matching requires unique, nonempty publications")
    payload = {"hypothesis": {k: candidate[k] for k in ("title", "text")},
               "publications": [{k: p[k] for k in ("paper_id", "title", "abstract")} for p in papers]}

    def validate(value):
        results = value["publications"]
        ids = [r["paper_id"] for r in results]
        if len(ids) != len(sources) or set(ids) != set(sources):
            raise ValueError("Must assess every publication exactly once")
        for result in results:
            paper = sources[result["paper_id"]]
            source = paper["title"] + "\n" + paper["abstract"]
            outcome = result["outcome"] in ("supporting", "contradictory", "mixed")
            if outcome and (not paper["abstract"].strip() or not result["source_passages"]):
                raise ValueError("Outcome evidence requires an abstract and exact passage")
            if any(not quote.strip() or quote not in source for quote in result["source_passages"]):
                raise ValueError("Quote absent from the identified publication")
        return value

    result = caller.call("assessment", identity, MATCH_SYSTEM +
                         " Assess each supplied publication independently. Return exactly one entry per paper_id."
                         " Never transfer evidence between publications. Keep rationales concise.",
                         payload, CorrespondenceBatch, validate)
    return [{**r, "rank": sources[r["paper_id"]]["rank"]} for r in result["publications"]]
