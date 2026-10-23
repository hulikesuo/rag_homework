"""Independent artifact-only final checks. Never loads models or calls APIs."""
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


def jl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = load_module("independent_main_metrics", "scripts/main_acceptance_audit.py")
    loader = load_module("final_loader", "src/loader.py")
    chunker = load_module("final_chunker", "src/chunker.py")
    docs = loader.read_all_docs(str(ROOT / "data/corpus"))
    corpus = {c["chunk_id"]: c for c in chunker.chunk_documents(docs)}
    qa = {q["id"]: q for q in jl(ROOT / "data/eval/qa_set.jsonl")}
    errors, pending, report = [], [], {}

    def check(ok, label):
        if not ok:
            errors.append(label)

    def evidence_ok(chunks, label):
        for c in chunks:
            orig = corpus.get(c.get("chunk_id"))
            check(orig is not None and c.get("text") == orig["text"], [label, "source_text", c.get("chunk_id")])
            if orig:
                for key in ("doc", "start", "end"):
                    if key in c:
                        check(c[key] == orig[key], [label, "source_span", key])

    checkpoint = OUT / "t3/t3_subquery_checkpoint.jsonl"
    summaries = {}
    success = []
    for strategy in ("baseline", "rewrite", "multi", "subquery"):
        path = OUT / f"t3/t3_cases_{strategy}.csv"
        if path.exists():
            rows = csv_rows(path)
        elif strategy == "subquery" and checkpoint.exists():
            rows = jl(checkpoint)
            pending.append("t3_subquery_csv")
        else:
            pending.append(f"t3_{strategy}")
            continue
        if len(rows) != 30:
            pending.append(f"t3_{strategy}_only_{len(rows)}_of_30")
        check(len(rows) == len({r["id"] for r in rows}), [strategy, "duplicate_ids"])
        for row in rows:
            tag = ["t3", strategy, row["id"]]
            q = qa[row["id"]]
            for k in ("level", "question", "gold_answer"):
                check(row[k] == q[k], [*tag, k])
            check(row["generation_status"] == row["query_status"] == "ok", [*tag, "status"])
            check(bool(row["answer"].strip()), [*tag, "nonempty_answer"])
            chunks = json.loads(row["retrieved_evidence"])
            evidence_ok(chunks, tag)
            ids = [c["chunk_id"] for c in chunks]
            check(ids == json.loads(row["retrieved"]) and len(ids) == len(set(ids)) == 5, [*tag, "top5"])
            gold = set(q["gold_chunk_ids"])
            hit = len(gold.intersection(ids[:5]))
            check(abs(float(row["Recall@5"]) - hit / len(gold)) < 1e-10, [*tag, "recall"])
            check(float(row["Hit@5"]) == float(hit > 0), [*tag, "hit"])
            check(abs(float(row["ROUGE-L_proxy"]) - base.rouge(q["gold_answer"], row["answer"])) < 1e-10, [*tag, "rouge"])
            normalize = lambda x: re.sub(r"\s+", "", x)
            check(float(row["EM_proxy"]) == float(normalize(q["gold_answer"]) == normalize(row["answer"])), [*tag, "em"])
            check(0 <= float(row["total_time_s"]) < 3600, [*tag, "elapsed_seconds"])
            if strategy == "subquery":
                subs = json.loads(row["subquery_answers"])
                union = json.loads(row["synthesis_context"])
                evidence_ok(union, [*tag, "union"])
                check(len(union) == int(row["synthesis_context_count"]), [*tag, "union_count"])
                check([c["chunk_id"] for c in union] == json.loads(row["synthesis_context_ids"]), [*tag, "union_ids"])
                expected = {(c["chunk_id"], c["text"]) for c in chunks}
                subhits = []
                queries = json.loads(row["queries"])
                check(bool(subs), [*tag, "subanswers_missing"])
                for sub in subs:
                    check(sub["query"] in queries and sub["query"] != q["question"], [*tag, "query_alignment"])
                    evidence_ok(sub["context"], [*tag, "sub_context"])
                    check(bool((sub.get("answer") or "").strip()), [*tag, "subanswer_empty"])
                    expected.update((c["chunk_id"], c["text"]) for c in sub["context"])
                    subhits.extend(c["chunk_id"] for c in sub["context"])
                check(expected == {(c["chunk_id"], c["text"]) for c in union}, [*tag, "evidence_union"])
                check(not json.loads(row["subquery_errors"]), [*tag, "subquery_errors"])
                if q["level"] in ("L2", "L3") and gold.intersection(subhits):
                    success.append({"id": row["id"], "subquery_gold_hits": sorted(gold.intersection(subhits)), "rrf_hit5": float(row["Hit@5"]), "union_size": len(union)})
        summaries[strategy] = {"n": len(rows), "hit5": statistics.mean(float(r["Hit@5"]) for r in rows), "recall5_fraction": statistics.mean(float(r["Recall@5"]) for r in rows), "rouge": statistics.mean(float(r["ROUGE-L_proxy"]) for r in rows)}
    baseline_path = OUT / "t3/t3_cases_baseline.csv"
    baseline_hits = {r["id"]: float(r["Hit@5"]) for r in csv_rows(baseline_path)} if baseline_path.exists() else {}
    success = [s for s in success if baseline_hits.get(s["id"]) == 0]
    if not success:
        pending.append("t3_real_l2_l3_failure_to_subquery_success_case")
    report["t3"] = {"strategies": summaries, "baseline_failure_to_subquery_evidence_success": success}

    control_dir = OUT / "t3/control_rrf"
    if (control_dir / "control_answers.csv").exists():
        answers = csv_rows(control_dir / "control_answers.csv")
        retrieval = csv_rows(control_dir / "control_retrieval.csv")
        raw_attempts = jl(control_dir / "control_answers.jsonl")
        raw_by_id = {r["id"]: r for r in raw_attempts}
        answer_by_id = {r["id"]: r for r in answers}
        check(len(answers) == len(answer_by_id) == len(retrieval) == 30, "t3_control_30_unique")
        check(set(answer_by_id) == set(qa), "t3_control_complete_ids")
        scores = []
        for row in retrieval:
            qid = row["id"]
            q, ans = qa[qid], answer_by_id[qid]
            check(ans["answer"] == raw_by_id[qid]["answer"], ["control", qid, "raw_answer"])
            check(bool(ans["answer"].strip()) and not ans["answer"].startswith("<error:"), ["control", qid, "answer_success"])
            ids = json.loads(row["retrieved"])
            check(ids == json.loads(ans["retrieved"]), ["control", qid, "retrieval_at_generation"])
            evidence = json.loads(row["retrieved_evidence"])
            evidence_ok(evidence, ["control", qid])
            check(ids == [c["chunk_id"] for c in evidence], ["control", qid, "evidence_ids"])
            n = len(set(q["gold_chunk_ids"]).intersection(ids[:5]))
            check(float(row["Hit@5"]) == float(n > 0), ["control", qid, "hit"])
            check(abs(float(row["Recall@5"]) - n / len(q["gold_chunk_ids"])) < 1e-10, ["control", qid, "recall"])
            scores.append(base.rouge(q["gold_answer"], ans["answer"]))
        report["t3"]["control_rrf30"] = {"n": len(answers), "raw_attempts": len(raw_attempts), "rouge": statistics.mean(scores), "hit5": statistics.mean(float(r["Hit@5"]) for r in retrieval), "recall5_fraction": statistics.mean(float(r["Recall@5"]) for r in retrieval)}
    else:
        pending.append("t3_control_answers")

    control20_dir = control_dir / "control20"
    if (control20_dir / "control_answers.csv").exists():
        answers20 = csv_rows(control20_dir / "control_answers.csv")
        comparison = csv_rows(OUT / "t3/control20_vs30.csv")
        comparison_by_id = {r["id"]: r for r in comparison}
        control20_raw = jl(control20_dir / "control_answers.jsonl")
        fresh_attempts = [r for r in control20_raw if r.get("source", "").startswith("fresh")]
        raw20_by_id = {r["id"]: r for r in control20_raw}
        fresh_by_id = {r["id"]: r for r in fresh_attempts}
        meta20 = json.loads((control20_dir / "meta.json").read_text(encoding="utf-8"))
        expected_fresh = {"q02", "q07", "q30"}
        check(len(answers20) == len({r["id"] for r in answers20}) == 30, "control20_30_unique")
        check(len(control20_raw) == len(raw20_by_id) == 30, "control20_raw_export_30_unique")
        check(len(comparison) == len(comparison_by_id) == 30 and set(comparison_by_id) == set(qa), "control20_comparison_ids")
        check(len(fresh_attempts) == 3 and set(fresh_by_id) == expected_fresh, "control20_exactly_three_fresh_attempts")
        check(meta20["fresh_api_calls"] == 3 and meta20["reused_rows"] == 27 and meta20["generation_error"] == 0, "control20_metadata_counts")
        hit20, recall20, rouge20 = [], [], []
        for row in answers20:
            qid = row["id"]
            q, cmp = qa[qid], comparison_by_id[qid]
            check(row["answer"] == raw20_by_id[qid]["answer"] and row["source"] == raw20_by_id[qid]["source"], ["control20", qid, "jsonl_export"])
            ids = json.loads(row["retrieved"])
            ids30 = json.loads(cmp["control30_ids"])
            check(ids == json.loads(cmp["control20_ids"]), ["control20", qid, "replayed_context"])
            check(ids30 == json.loads(answer_by_id[qid]["retrieved"]), ["control20", qid, "control30_context"])
            check(len(ids) == len(set(ids)) == 5 and all(cid in corpus for cid in ids), ["control20", qid, "valid_source_chunks"])
            check(row["context_hash"] == hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode("utf-8")).hexdigest(), ["control20", qid, "context_hash"])
            check(row["question"] == q["question"] and row["level"] == q["level"], ["control20", qid, "question"])
            check(row["generation_status"] == "ok" and not row["generation_error"] and bool(row["answer"].strip()) and not row["answer"].startswith("<error:"), ["control20", qid, "successful_answer"])
            if qid in expected_fresh:
                check(ids != ids30 and row["source"] == "fresh_api", ["control20", qid, "fresh_context"])
                check(row["answer"] == fresh_by_id[qid]["answer"] and row["retrieved"] == fresh_by_id[qid]["retrieved"], ["control20", qid, "raw_attempt"])
            else:
                check(ids == ids30 and row["source"] == "reuse_control30" and row["answer"] == answer_by_id[qid]["answer"], ["control20", qid, "exact_context_and_answer_reuse"])
            n = len(set(q["gold_chunk_ids"]).intersection(ids))
            hit20.append(float(n > 0))
            recall20.append(n / len(q["gold_chunk_ids"]))
            rouge20.append(base.rouge(q["gold_answer"], row["answer"]))
            check(float(cmp["control20_Hit@5"]) == hit20[-1] and abs(float(cmp["control20_Recall@5"]) - recall20[-1]) < 1e-10, ["control20", qid, "retrieval_metrics"])
        report["t3"]["control_rrf20"] = {"n": len(answers20), "fresh_api_answers": len(fresh_attempts), "reused_identical_context_answers": 27, "rouge": statistics.mean(rouge20), "hit5": statistics.mean(hit20), "recall5_fraction": statistics.mean(recall20)}
        summary20 = next(r for r in csv_rows(OUT / "t3/t3_summary.csv") if r["strategy"] == "control_original_rrf20")
        for column, metric in (("Hit@5", "hit5"), ("Recall@5", "recall5_fraction"), ("ROUGE-L_proxy", "rouge")):
            check(abs(float(summary20[column]) - report["t3"]["control_rrf20"][metric]) < 0.0001, ["control20", "summary", column])
    else:
        pending.append("t3_control20_matched_budget_answers")

    alpha_path = OUT / "t3/t3_alpha_per_question.csv"
    if alpha_path.exists():
        rows = csv_rows(alpha_path)
        groups = {a: [r for r in rows if float(r["alpha_dense"]) == a] for a in (0, .3, .5, .7, 1)}
        alpha_summary = csv_rows(OUT / "t3/t3_linear_alpha_sweep.csv")
        check(len(rows) == 150 and len(alpha_summary) == 5, "t3_alpha_grid_150")
        for a, subset in groups.items():
            check(len(subset) == 30 and {r["id"] for r in subset} == set(qa), ["alpha", a, "ids"])
            for row in subset:
                ids, gold = json.loads(row["retrieved"]), set(qa[row["id"]]["gold_chunk_ids"])
                n = len(gold.intersection(ids[:5]))
                check(len(ids) == len(set(ids)) == 5 and all(c in corpus for c in ids), ["alpha", a, row["id"], "top5"])
                check(float(row["hit_at_5"]) == float(n > 0) and abs(float(row["recall_at_5"]) - n / len(gold)) < 1e-10, ["alpha", a, row["id"], "metrics"])
            s = next(r for r in alpha_summary if float(r["alpha_dense"]) == a)
            check(abs(float(s["Hit@5"]) - statistics.mean(float(r["hit_at_5"]) for r in subset)) < 1e-10, ["alpha", a, "mean_hit"])
            check(abs(float(s["Recall@5"]) - statistics.mean(float(r["recall_at_5"]) for r in subset)) < 1e-10, ["alpha", a, "mean_recall"])
        report["t3"]["alpha_grid"] = [{"alpha": float(r["alpha_dense"]), "course_hit5": float(r["Hit@5"]), "fraction_recall5": float(r["Recall@5"])} for r in alpha_summary]
    else:
        pending.append("t3_alpha_grid")

    judge = OUT / "t7/t7_llm_judge.csv"
    cp = OUT / "t7/judge_checkpoint.jsonl"
    if judge.exists() and cp.exists():
        attempts = jl(cp)
        latest = {r["id"]: r for r in attempts}
        rows = csv_rows(judge)
        for row in rows:
            raw = latest[row["id"]]
            for k in ("raw_output", "judge_messages_json", "answer_hash", "context_hash"):
                check(row[k] == raw[k], ["t7", row["id"], "batch_export", k])
        report["t7_batch_selection"] = {"all_attempts": len(attempts), "selected": len(rows), "duplicate_id_attempts": {k: v for k, v in Counter(r["id"] for r in attempts).items() if v > 1}, "claims": sum(int(r["total_claims"]) for r in rows)}
        worst_path = OUT / "t7/04_worst5_provenance.json"
        if worst_path.exists():
            worst = json.loads(worst_path.read_text(encoding="utf-8"))["worst5"]
            check({r["id"] for r in worst} == {"q01", "q22", "q28", "q02", "q13"}, "t7_fixed_worst5")
            t4 = {r["id"]: r for r in json.loads((ROOT / "results/t4_per_query.json").read_text(encoding="utf-8"))}
            t1 = {r["id"]: r for r in jl(ROOT / "results/t1_baseline.jsonl")}
            for row in worst:
                qid = row["id"]
                check(row["t4_hybrid_top10_ids"] == t4[qid]["modes"]["hybrid"]["top10_ids"], ["worst5", qid, "top10"])
                check(row["actual_top5_ids"] == t1[qid]["retrieved"], ["worst5", qid, "actual_top5"])
    else:
        pending.append("t7_batch_export")

    http = OUT / "demo/http_e2e_results.json"
    if http.exists():
        data = json.loads(http.read_text(encoding="utf-8"))
        cases = data["real_e2e"]
        check(len(cases) >= 2, "demo_two_http_cases")
        for row in cases:
            tag = ["demo", row["case"]]
            resp = row["response"]
            check(row["ok"] and not row["err_type"] and row["elapsed_sec"] > 0, [*tag, "request_success"])
            check(bool(resp["answer"].strip()) and "生成失败" not in resp["answer"], [*tag, "answer"])
            check(bool(resp["evidence"].strip()) and bool(resp.get("stats")), [*tag, "evidence_stats"])
            check(row["request"]["mode"] in ("dense", "bm25", "hybrid"), [*tag, "mode"])
        report["demo"] = {"http_cases": len(cases), "transport": data["transport"], "browser_visual_verified": False}
    else:
        pending.append("demo_http_e2e")

    changed = [p for p, h in base.formal_hashes().items() if not (ROOT / p).exists() or sha(ROOT / p) != h]
    check(not changed, ["formal_results_changed", changed])
    report.update(created_at=datetime.now(timezone.utc).isoformat(), errors=errors, pending=pending, scope="Artifact-only T3 metrics/traces, T7 batch selection, Demo HTTP; packaging/report separately verified")
    (OUT / "main_final_artifact_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
