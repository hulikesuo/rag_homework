"""Print read-only reproduction anchors without exposing service secrets.

The script reads only the formal question set, result metadata, and installed
package versions. It never opens .env, never calls an API, and never writes
outside an explicitly requested output path.
"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ANCHOR_FILES = [
    "data/eval/qa_set.jsonl",
    "results/t1_baseline.jsonl",
    "results/t2_chunking.csv",
    "results/t2_chunking_cases.csv",
    "results/t5_rerank.csv",
    "results/t7_llm_judge.csv",
]

# Names come from requirements.txt plus their direct runtime dependencies.
PACKAGES = [
    "sentence-transformers",
    "faiss-cpu",
    "rank-bm25",
    "langchain-core",
    "openai",
    "brotlicffi",
    "python-docx",
    "pandas",
    "matplotlib",
    "gradio",
    "numpy",
    "torch",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "httpx",
    "httpx2",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_from_results():
    path = os.path.join("results", "t1_baseline.jsonl")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                for key in ("generator_model", "model", "llm_model"):
                    if row.get(key):
                        return str(row[key])
                break
    path = os.path.join("results", "t2_chunking.csv")
    if os.path.exists(path):
        import csv
        with open(path, encoding="utf-8-sig", newline="") as handle:
            row = next(csv.DictReader(handle), {})
        return row.get("generator_model") or None
    return None


def _optional_package_notes():
    notes = {
        "brotlicffi": (
            "Not a runtime dependency on CPython: httpx2's optional 'brotli' "
            "extra requires brotli on CPython and brotlicffi only on non-CPython."
        ),
        "pypdf": (
            "Optional PDF utility, not required for the DOCX corpus; delivery "
            "PDF verification uses bundled PDF tooling."
        ),
    }
    try:
        from openai import _httpx
        notes["openai_transport"] = getattr(_httpx, "__name__", "openai._httpx")
    except Exception as exc:
        notes["openai_transport"] = f"unavailable: {type(exc).__name__}"
    return notes


def collect():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    anchors = {}
    for relative in ANCHOR_FILES:
        anchors[relative] = {
            "exists": os.path.isfile(relative),
            "sha256": sha256(relative) if os.path.isfile(relative) else None,
        }
    from scripts.t5_delivery_figure import BOOTSTRAP_SEED
    return {
        "python": {
            "executable_name": os.path.basename(sys.executable),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "model": _model_from_results() or "待核实（结果文件未给出实际模型）",
        "seed": {
            "t5_bootstrap": BOOTSTRAP_SEED,
            "note": (
                "固定种子与 temperature 不能保证生成 API 完全确定性；"
                "实际生成模型、题集 hash、结果 hash 才是复现主锚点。"
            ),
        },
        "packages": packages,
        "packages_not_installed_optional": [
            name for name, version in packages.items() if version is None
        ],
        "optional_package_notes": _optional_package_notes(),
        "anchors": anchors,
        "secrets_read": False,
        "network_used": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="optional JSON output path")
    parser.add_argument("--check", action="store_true",
                        help="fail when a formal anchor is missing")
    args = parser.parse_args()
    report = collect()
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"[reproduce] wrote {os.path.abspath(args.output)}")
    else:
        print(text)
    if args.check and any(not item["exists"] for item in report["anchors"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
