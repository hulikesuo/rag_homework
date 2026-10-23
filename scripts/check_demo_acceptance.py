"""Demo acceptance checks.

Phase 1 runs real L1/L4 against a localhost Gradio server through
gradio_client and stores full response JSON.  If no server is already
running on 127.0.0.1:7860 it starts one and stops only that child after
the checks.  Edge/error inputs are covered by offline stub tests in
scripts/check_demo_acceptance_stub_test.py; this file does not rely on
browser UI automation and does not claim visual inspection.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "repair_outputs" / "acceptance_20261023" / "demo"
URL = "http://127.0.0.1:7860/"
HOST = "127.0.0.1"
PORT = 7860
PYTHON = sys.executable


def _append_log(lines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "http_e2e.log").open("a", encoding="utf-8") as h:
        h.write("\n".join(lines) + "\n")


def _port_listening() -> tuple[bool, int]:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((HOST, PORT))
            return True, 0
        except OSError:
            return False, 0


def _start_server() -> subprocess.Popen:
    env = dict(__import__("os").environ)
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": str(ROOT),
    })
    log_path = OUT / "http_server.out.txt"
    log_handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [PYTHON, "-X", "utf8", "-B", "-m", "demo.app"],
        cwd=str(ROOT), env=env, stdout=log_handle, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    _append_log([f"STARTED_CHILD pid={proc.pid} url={URL}"])
    for _ in range(120):
        if _port_listening()[0]:
            return proc
        if proc.poll() is not None:
            _append_log([
                f"SERVER_EXITED pid={proc.pid} rc={proc.returncode}",
                "---server log tail---",
                "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:]),
            ])
            raise RuntimeError("demo.app exited before listening")
        time.sleep(0.5)
    _append_log(["SERVER_START_TIMEOUT_NO_LISTENER"])
    raise RuntimeError("demo.app did not listen on 127.0.0.1:7860")


def _load_qas() -> dict[str, dict]:
    qas = {}
    for line in (ROOT / "data/eval/qa_set.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            qas[q["id"]] = q
    from pathlib import Path as _P
    picks = [_P(x) for x in []]
    del picks
    return qas


def _pick_question(qas: dict[str, dict], level: str, fallback: str) -> str:
    for q in qas.values():
        if q.get("level") == level:
            return q["question"]
    return fallback


def _run_real(client, qas: dict[str, dict], level: str,
               fallback: str, mode: str, do_compress: bool, top_k: int) -> dict:
    question = _pick_question(qas, level, fallback)
    t0 = time.time()
    try:
        pred = client.predict(
            question, mode, do_compress, top_k, api_name="/answer_fn")
        ok = True
        err = ""
    except Exception as exc:
        pred = ["", "", ""]
        ok = False
        err = type(exc).__name__
    elapsed = round(time.time() - t0, 3)
    result = {
        "case": f"{level}_real",
        "ok": ok,
        "err_type": err,
        "elapsed_sec": elapsed,
        "request": {
            "question": question,
            "mode": mode,
            "do_compress": do_compress,
            "top_k": top_k,
        },
        "response": {
            "answer": (pred[0] if isinstance(pred, (list, tuple)) and pred
                       else ""),
            "evidence": (pred[1] if isinstance(pred, (list, tuple))
                         and len(pred) > 1 else ""),
            "stats": (pred[2] if isinstance(pred, (list, tuple))
                      and len(pred) > 2 else ""),
        },
    }
    summary = {
        "case": result["case"],
        "ok": result["ok"],
        "err_type": result["err_type"],
        "elapsed_sec": result["elapsed_sec"],
        "answer_len": len(result["response"]["answer"]),
        "evidence_len": len(result["response"]["evidence"]),
        "stats_len": len(result["response"]["stats"]),
    }
    _append_log([
        f"[{level}] ok={ok} elapsed={elapsed} err={err} "
        f"ans={summary['answer_len']} ev={summary['evidence_len']} "
        f"stats={summary['stats_len']}",
        json.dumps(summary, ensure_ascii=False),
    ])
    return result


def main() -> int:
    from gradio_client import Client

    OUT.mkdir(parents=True, exist_ok=True)
    already = _port_listening()[0]
    proc = None
    if not already:
        proc = _start_server()
    client = Client(URL, verbose=False)

    qas = _load_qas()
    results = [
        _run_real(client, qas, "L1",
                  "学生在校期间平均绩点怎么计算？", "hybrid", False, 5),
        _run_real(client, qas, "L4",
                  "请依据培养方案给出跨文化沟通的调整建议。",
                  "hybrid", True, 5),
    ]

    payload = {
        "server_url": URL,
        "transport": "gradio_client http queue",
        "api_name": "/answer_fn",
        "real_e2e": results,
        "edge_or_error_covered_in": "scripts/check_demo_acceptance_stub_test.py",
        "visual_inspection": "not performed",
    }
    with (OUT / "http_e2e_results.json").open(
            "w", encoding="utf-8") as h:
        json.dump(payload, h, ensure_ascii=False, indent=2)

    ok = all(r["ok"] for r in results)
    _append_log([f"HTTP_E2E_RESULT: {ok}"])
    print("OK" if ok else "FAIL", json.dumps(
        [{"case": r["case"], "ok": r["ok"], "err": r["err_type"],
          "answer_len": len(r["response"]["answer"]),
          "evidence_len": len(r["response"]["evidence"]),
          "stats_len": len(r["response"]["stats"])} for r in results],
        ensure_ascii=False))
    if proc is not None:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        _append_log([f"STOPPED_CHILD pid={proc.pid}"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
