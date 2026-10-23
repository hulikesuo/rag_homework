"""Independent, no-model audit of isolated acceptance artifacts."""
import csv
import hashlib
import importlib.util
import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "repair_outputs/acceptance_20261023"
if not OUT.is_dir():
    OUT = ROOT / "results/acceptance_20261023"


def formal_hashes():
    original = OUT / "orchestration_baseline.json"
    if original.exists():
        return json.loads(original.read_text(encoding="utf-8"))["formal_result_hashes"]
    snapshot = json.loads((ROOT / "delivery/formal_results_snapshot.json").read_text(encoding="utf-8"))
    return {"results/" + p: (v["sha256"] if isinstance(v, dict) else v) for p, v in snapshot["files"].items()}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def obj_sha(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def merged(spans):
    result = []
    for start, end in sorted(spans):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def coverage(gold, retrieved):
    total = covered = 0
    for doc in {g["doc"] for g in gold}:
        gs = merged([(g["start"], g["end"]) for g in gold if g["doc"] == doc])
        rs = merged([(c["start"], c["end"]) for c in retrieved if c["doc"] == doc])
        total += sum(b - a for a, b in gs)
        covered += sum(max(0, min(b, y) - max(a, x)) for a, b in gs for x, y in rs)
    return covered / total if total else 0


def rouge(gold, answer):
    # Independent LCS implementation; the project uses character-level ROUGE-L.
    a, b = gold or "", answer or ""
    if not a or not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for ca in a:
        current = [0]
        for j, cb in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if ca == cb else max(current[-1], previous[j]))
        previous = current
    return 2 * previous[-1] / (len(a) + len(b))


def main():
    errors, pending, report = [], [], {}

    def check(ok, item):
        if not ok:
            errors.append(item)

    changed = [p for p, h in formal_hashes().items() if not (ROOT / p).exists() or sha(ROOT / p) != h]
    check(not changed, {"formal_results_changed": changed})
    check(sha(ROOT / "data/eval/qa_set.jsonl") == "bfcbefc1ba2519c9728451559d30fd55f05151f903a927d836011ba07ae7b969", "formal_qa_changed")
    loader = module("independent_loader", "src/loader.py")
    chunker = module("independent_chunker", "src/chunker.py")
    docs = loader.read_all_docs(str(ROOT / "data/corpus"))
    chunks = {c["chunk_id"]: c for c in chunker.chunk_documents(docs)}
    qas = {q["id"]: q for q in read_jsonl(ROOT / "data/eval/qa_set.jsonl")}

    path = OUT / "t2/t2_advanced_chunking_cases_final.csv"
    if not path.exists():
        path = OUT / "t2/t2_advanced_chunking_cases.csv"
    partial = not path.exists()
    if partial:
        path = OUT / "t2/_tmp/checkpoint_case.csv"
        pending.append("t2_final")
    if path.exists():
        rows = read_csv(path)
        counts = Counter()
        for row in rows:
            key = (row["method"], int(row["size_for_index"]))
            counts[key] += 1
            tag = ["t2", *key, row["id"]]
            q = qas[row["id"]]
            for field in ("question", "gold_answer", "level"):
                check(row[field] == q[field], [*tag, field])
            check(row["retrieval_status"] == row["generation_status"] == "ok", [*tag, "status"])
            gold = [chunks[c] for c in q["gold_chunk_ids"]]
            small = json.loads(row["retrieved_small_evidence"] or "[]")
            parents = json.loads(row["retrieved_parent_evidence"] or "[]")
            for c in small + parents:
                check(c["text"] == docs[c["doc"]][c["start"]:c["end"]], [*tag, "span", c["chunk_id"]])
            for c in small:
                check(len(c["text"]) <= key[1], [*tag, "chunk_size"])
            for prefix, cs in (("small", small), ("parent", parents)):
                if prefix == "parent" and key[0] != "small_to_big":
                    continue
                actual = coverage(gold, cs[:5])
                hit = float(any(g["doc"] == c["doc"] and min(g["end"], c["end"]) > max(g["start"], c["start"]) for g in gold for c in cs[:5]))
                check(abs(actual - float(row[f"Recall@5_{prefix}_evidence"])) < 1e-10, [*tag, prefix, "coverage"])
                check(hit == float(row[f"Hit@5_{prefix}_evidence"]), [*tag, prefix, "hit"])
            if key[0] == "small_to_big":
                pm = {p["chunk_id"]: p for p in parents}
                for c in small:
                    p = pm.get(c["parent_chunk_id"])
                    check(p is not None and p["start"] <= c["start"] < c["end"] <= p["end"], [*tag, "parent_contains_child"])
                    check(p is not None and c["chunk_id"] in p.get("parent_of", []), [*tag, "parent_of"])
            if row["generation_status"] == "ok":
                check(bool(row["answer"].strip()), [*tag, "empty_answer"])
                check(abs(rouge(q["gold_answer"], row["answer"]) - float(row["ROUGE-L_proxy"])) < 1e-9, [*tag, "rouge"])
        expected = {(m, s) for m in ("semantic", "small_to_big") for s in (256, 512, 1024)}
        if not partial:
            check(set(counts) == expected and all(n == 30 for n in counts.values()), "t2_config_completeness")
        report["t2"] = {"partial": partial, "rows": len(rows), "config_counts": {str(k): v for k, v in counts.items()}}
        if "_final" in path.name:
            raw = read_csv(path.parent / "t2_advanced_chunking_cases.csv")
            key = lambda r: (r["id"], r["method"], r["size_for_index"])
            originals = {key(r): r for r in raw}
            changed_keys = []
            for row in rows:
                original = originals[key(row)]
                if original["generation_status"] == "ok":
                    check(all(row[k] == original[k] for k in original), ["t2", key(row), "unnecessary_retry"])
                else:
                    changed_keys.append(key(row))
                    for field in ("question", "gold_answer", "retrieved_small_evidence", "retrieved_parent_evidence"):
                        check(original[field] == row[field], ["t2", key(row), "retry_context_changed", field])
            report["t2"]["recovered_error_rows"] = changed_keys

    path = OUT / "t7/t7_llm_judge.csv"
    if path.exists():
        rows = read_csv(path)
        source = ROOT / "results/t1_baseline.jsonl"
        records = {r["id"]: r for r in read_jsonl(source)}
        check(len(rows) == len({r["id"] for r in rows}) == 30 and {r["id"] for r in rows} == set(qas), "t7_ids")
        for row in rows:
            q, rec = qas[row["id"]], records[row["id"]]
            tag = ["t7", row["id"]]
            check(row["source_cache_sha"] == sha(source), [*tag, "source_hash"])
            check(row["answer_hash"] == hashlib.sha256(rec["answer"].encode()).hexdigest(), [*tag, "answer_hash"])
            check(row["context_hash"] == obj_sha(rec["context_texts"]), [*tag, "context_hash"])
            check(row["level"] == q["level"] and row["judge_status"] == "ok", [*tag, "level_status"])
            u, n, f = int(row["unsupported_claims"]), int(row["total_claims"]), int(row["faithfulness_1_5"])
            check(0 <= u <= n and n > 0 and (f != 5 or u == 0), [*tag, "contradiction"])
            check(abs(float(row["hallucination_rate"]) - u / n) < 0.00006, [*tag, "hallucination_rate"])
            messages = json.loads(row["judge_messages_json"])
            payload = json.loads(messages[1]["content"].split("\n", 1)[1])
            check(payload["question"] == q["question"] and payload["gold_answer_or_rubric"] == q["gold_answer"] and payload["candidate_answer"] == rec["answer"] and payload["actual_context"] == rec["context_texts"], [*tag, "prompt_payload"])
            check(row["prompt_hash"] == obj_sha({"version": int(row["prompt_version"]), "messages": messages}), [*tag, "prompt_hash"])
            parsed = json.loads(row["raw_output"].strip().removeprefix("```json").removesuffix("```").strip())
            for k in ("faithfulness_1_5", "unsupported_claims", "total_claims", "correctness_0_5"):
                check(int(row[k]) == parsed[k], [*tag, "raw_score", k])
        summary = json.loads((path.parent / "t7_llm_judge_summary.json").read_text(encoding="utf-8"))
        check(summary["n"] == summary["ok"] == 30 and summary["errors"] == 0, "t7_summary_count")
        for lv in ("L1", "L2", "L3", "L4", "ALL"):
            sub = rows if lv == "ALL" else [r for r in rows if r["level"] == lv]
            s = summary["levels"][lv]
            u, n = sum(int(r["unsupported_claims"]) for r in sub), sum(int(r["total_claims"]) for r in sub)
            check(s["n"] == len(sub) and s["unsupported_total"] == u and s["claims_total"] == n, ["t7", lv, "sum"])
            check(abs(s["hallucination_rate_micro"] - u / n) < 0.00006, ["t7", lv, "micro"])
            for field, output in (("faithfulness_1_5", "faithfulness_1_5_mean"), ("correctness_0_5", "correctness_0_5_mean")):
                check(abs(statistics.mean(float(r[field]) for r in sub) - s[output]) < 0.00006, ["t7", lv, output])
        report["t7"] = {"rows": len(rows), "unsupported": summary["levels"]["ALL"]["unsupported_total"], "claims": summary["levels"]["ALL"]["claims_total"]}
    else:
        pending.append("t7_final")

    path = OUT / "t4/t4_acceptance_per_query.json"
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
        counts = Counter()
        for row in rows:
            gold = set(row["gold_chunk_ids"])
            for mode, data in row["modes"].items():
                hits = data["top10"]
                ids = [c["chunk_id"] for c in hits]
                check(len(ids) == len(set(ids)) == 10, ["t4", row["id"], mode, "top10_ids"])
                for c in hits:
                    original = chunks.get(c["chunk_id"])
                    check(original is not None and all(c[k] == original[k] for k in ("text", "doc", "start", "end")), ["t4", row["id"], mode, "corpus_evidence"])
                for k in (1, 3, 5, 10):
                    n = len(gold.intersection(ids[:k]))
                    check(data[f"Hit@{k}"] == float(n > 0), ["t4", row["id"], mode, "hit", k])
                    check(abs(data[f"Recall@{k}"] - n / len(gold)) < 1e-12, ["t4", row["id"], mode, "recall", k])
                mrr = next((1 / i for i, cid in enumerate(ids, 1) if cid in gold), 0)
                check(abs(mrr - data["MRR@10"]) < 1e-12, ["t4", row["id"], mode, "mrr"])
                check([c["chunk_id"] for c in data["top3"]] == ids[:3], ["t4", row["id"], mode, "top3"])
            dense, sparse = row["modes"]["dense"], row["modes"]["bm25"]
            a, b = (dense["Recall@3"], dense["MRR@10"]), (sparse["Recall@3"], sparse["MRR@10"])
            winner = "dense" if a > b else "bm25" if b > a else "tie"
            counts[winner] += 1
            check(row["analysis"]["winner"] == winner, ["t4", row["id"], "winner"])
        report["t4"] = {"rows": len(rows), "strict_winners": dict(counts), "focused_batch_alone_meets_targets": counts["dense"] >= 5 and counts["bm25"] >= 5}
        selected_path = path.parent / "t4_case_10.json"
        if selected_path.exists():
            selected = json.loads(selected_path.read_text(encoding="utf-8"))
            formal = json.loads((ROOT / "results/t4_per_query.json").read_text(encoding="utf-8"))
            source = {(group, r["id"]): r for group, items in (("formal30", formal), ("exploratory50", rows)) for r in items}
            wins = Counter()
            for row in selected:
                group = row["source_group"]
                if group != "formal30":
                    group = "exploratory50"
                original = source.get((group, row["id"]))
                check(original is not None, ["t4_selected", row["id"], "source"])
                if original is None:
                    continue
                check(row["question"] == original["question"] and row["gold_chunk_ids"] == original["gold_chunk_ids"], ["t4_selected", row["id"], "qa_alignment"])
                for mode in ("dense", "bm25"):
                    check([c["chunk_id"] for c in row[f"{mode}_top3"]] == [c["chunk_id"] for c in original["modes"][mode]["top3"]], ["t4_selected", row["id"], mode, "top3"])
                a, b = [(original["modes"][mode]["Recall@3"], original["modes"][mode]["MRR@10"]) for mode in ("dense", "bm25")]
                winner = "dense" if a > b else "bm25" if b > a else "tie"
                check(winner == row["winner_strict_dense_vs_bm25"] and winner != "tie", ["t4_selected", row["id"], "strict_winner"])
                wins[winner] += 1
            check(len(selected) == 10 and wins["dense"] == wins["bm25"] == 5, "t4_selected_5_plus_5")
            report["t4"]["selected_valid_strict_case_counts"] = dict(wins)
    else:
        pending.append("t4_final")

    report.update(created_at=datetime.now(timezone.utc).isoformat(), errors=errors, pending=pending, formal_results_changed=changed)
    (OUT / "main_new_results_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
