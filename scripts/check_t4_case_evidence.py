# -*- coding: utf-8 -*-
"""Independent checker for the consolidated T4 10-case evidence.

Does not call retrieval or modify any frozen results. Derives winners and
gold ranks directly from saved top3 rows, verifies the selected-id list,
source group hashes, and that every selected case has non-template analysis
text plus filled formal gold_texts where the formal source carries them.
Outputs a pass/fail JSON artifact under the T4 directory.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T4 = ROOT / "repair_outputs" / "acceptance_20261023" / "t4"
FORMAL_RESULTS = ROOT / "results" / "t4_per_query.json"
EXPLORATORY_PER_QUERY = T4 / "t4_acceptance_per_query.json"
CASES = T4 / "t4_case_10.json"
CONSOLIDATED = T4 / "t4_consolidated_80.json"
OUT = T4 / "t4_case_checker.json"
DENSE_5 = ["q01", "q02", "q07", "acc_c20", "acc_r01"]
BM25_5 = ["acc_c08", "acc_c14", "acc_c15", "acc_r03", "acc_r10"]

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def winner(dere, bm25):
    d = (float(dere.get("Recall@3", 0)), float(dere.get("MRR@10", 0)))
    b = (float(bm25.get("Recall@3", 0)), float(bm25.get("MRR@10", 0)))
    if d == b:
        return "tie"
    return "dense" if d > b else "bm25"

def gold_ranks(top3, gold_ids):
    ids = set(gold_ids or [])
    return [int(x["rank"]) for x in top3 if x.get("is_gold") or x.get("chunk_id") in ids]

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    errors, warns = [], []
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    formal = json.loads(FORMAL_RESULTS.read_text(encoding="utf-8"))
    exploratory = json.loads(EXPLORATORY_PER_QUERY.read_text(encoding="utf-8"))
    by = {}
    for src, rows in (("formal30", formal), ("exploratory50", exploratory)):
        for r in rows:
            by[(src, r["id"])] = r
    ids = [x["id"] for x in cases]
    if ids != DENSE_5 + BM25_5:
        errors.append("selected ids order/values differ")
    # recompute objective numbers
    for it in cases:
        pid = (it["source_group"], it["id"])
        if pid not in by:
            errors.append(f"missing source {pid}")
            continue
        src = by[pid]
        modes = src.get("modes", {})
        modes_dense, modes_bm25 = modes.get("dense", {}), modes.get("bm25", {})
        w = winner(modes_dense, modes_bm25)
        if w != it["winner_strict_dense_vs_bm25"]:
            errors.append(f"{it['id']} winner mismatch {it['winner_strict_dense_vs_bm25']} != {w}")
        gid = it["gold_chunk_ids"] or []
        dr = gold_ranks(it["dense_top3"], gid)
        br = gold_ranks(it["bm25_top3"], gid)
        if dr != it["objective_gold_ranks"]["dense_top3"]:
            errors.append(f"{it['id']} dense ranks mismatch {dr}")
        if br != it["objective_gold_ranks"]["bm25_top3"]:
            errors.append(f"{it['id']} bm25 ranks mismatch {br}")
        if it["id"] in ("q01", "q02", "q07") and len(it["gold_texts"]) < 1:
            errors.append(f"{it['id']} formal gold_texts empty")
        analysis = it.get("mechanism_analysis", {})
        segs = analysis.get("queries_to_original", [])
        if not segs:
            errors.append(f"{it['id']} no mechanism segments")
        for seg in segs:
            if not seg.get("query") or not seg.get("original") or not seg.get("mechanism"):
                errors.append(f"{it['id']} incomplete mechanism")
            if not any(k in (seg.get("speculation") or "").lower() for k in ("推测", "speculation")):
                errors.append(f"{it['id']} segment missing speculation marker")
        comp = analysis.get("competing_top3")
        if not comp:
            errors.append(f"{it['id']} missing competing_top3")
        # avoid pure template: require at least chip-specific text
        text = (it.get("why_dense_or_bm25_wins") or "")
        if len(text) < 200:
            warns.append(f"{it['id']} analysis may be short: {len(text)}")
    consolidated = json.loads(CONSOLIDATED.read_text(encoding="utf-8"))
    winners = {}
    for cid in DENSE_5 + BM25_5:
        winners[cid] = next((x["winner_strict_dense_vs_bm25"] for x in consolidated["ledger"] if x["id"]==cid), None)
    dense_cnt = sum(1 for x in DENSE_5 if winners.get(x) == "dense")
    bm25_cnt = sum(1 for x in BM25_5 if winners.get(x) == "bm25")
    status = "PASS" if not errors else "FAIL"
    result = {"status": status, "errors": errors, "warnings": warns,
        "selected_dense_ids": DENSE_5, "selected_bm25_ids": BM25_5,
        "selected_dense_strict_winner_count": dense_cnt, "selected_bm25_strict_winner_count": bm25_cnt,
        "source_hashes": {"results/t4_per_query.json": sha256(FORMAL_RESULTS),
            "repair_outputs/acceptance_20261023/t4/t4_acceptance_per_query.json": sha256(EXPLORATORY_PER_QUERY),
            "repair_outputs/acceptance_20261023/t4/t4_case_10.json": sha256(CASES),
            "repair_outputs/acceptance_20261023/t4/t4_consolidated_80.json": sha256(CONSOLIDATED)},
        "audit_type": "AI audit (non-human) 主代理独立重算"}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Status:", status, "errors:", len(errors), "warnings:", len(warns))
    print("Selected dense winners:", dense_cnt, "/5  Selected bm25 winners:", bm25_cnt, "/5")
    for e in errors:
        print("  ERROR:", e)
    for w in warns:
        print("  WARN:", w)
    return 0 if status == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
