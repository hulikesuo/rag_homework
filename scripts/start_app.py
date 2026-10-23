#!/usr/bin/env python3
"""Launch the local Gradio UI or Docker Compose without publishing local inputs."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("gradio", "sentence_transformers", "faiss", "rank_bm25", "docx",
           "openai", "langchain_core", "numpy", "requests")
EXTENSIONS = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf", ".docx"}


def read_env(path):
    values = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key.startswith("export "):
                key = key[7:].strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                values[key] = value
    return values


def runtime_env(root, environ, args):
    env = read_env(root / ".env")
    env.update(environ)
    if args.offline:
        env.update(RAG_OFFLINE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.host:
        env["RAG_BIND_ADDRESS" if args.docker else "RAG_HOST"] = args.host
    if args.port is not None:
        env["RAG_PORT"] = str(args.port)
    env.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    return env


def validate_inputs(root, env, docker=False):
    errors = []
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        value = env.get(key, "").strip()
        if not value or value.lower().startswith(("your_", "your-", "replace_", "<")):
            errors.append(f"Set {key} in .env (or your environment).")
    url = urlsplit(env.get("LLM_BASE_URL", ""))
    if url.scheme not in {"http", "https"} or not url.hostname:
        errors.append("LLM_BASE_URL must be an HTTP(S) OpenAI-compatible base URL.")
    if url.username or url.password:
        errors.append("Do not put credentials in LLM_BASE_URL; use LLM_API_KEY.")
    if docker and url.hostname in {"localhost", "127.0.0.1", "::1"}:
        errors.append("For a host-side LLM in Docker, use host.docker.internal instead of localhost.")
    corpus = root / "data" / "corpus" if docker else Path(
        env.get("RAG_CORPUS_DIR", str(root / "data" / "corpus")))
    if not corpus.is_absolute():
        corpus = root / corpus
    if not corpus.is_dir() or not any(
            p.is_file() and p.stat().st_size > 0 and p.suffix.lower() in EXTENSIONS
            and not p.name.startswith("~$") for p in corpus.rglob("*")):
        errors.append("Prepare at least one non-empty supported document in data/corpus (or RAG_CORPUS_DIR).")
    try:
        port = int(env.get("RAG_PORT", "7860"))
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        errors.append("RAG_PORT must be an integer between 1 and 65535.")
    host = env.get("RAG_BIND_ADDRESS" if docker else "RAG_HOST", "127.0.0.1")
    if not host or any(c.isspace() for c in host) or "/" in host or "://" in host:
        errors.append("Bind address must be a hostname or IP address, not a URL.")
    return errors


def run(command, env, root=ROOT):
    return subprocess.run(command, cwd=root, env=env, check=True)


def local_python(root, env, no_install):
    if no_install:
        missing = [name for name in MODULES if importlib.util.find_spec(name) is None]
        if missing:
            raise RuntimeError("Missing packages: " + ", ".join(missing) +
                               ". Activate the correct environment or omit --no-install.")
        return sys.executable
    venv = root / ".venv"
    interpreter = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not interpreter.is_file():
        print("Creating .venv (local files and .env are preserved).", flush=True)
        run([sys.executable, "-m", "venv", str(venv)], env, root)
    marker = venv / ".rag-requirements.sha256"
    import hashlib
    fingerprint = hashlib.sha256((root / "requirements.txt").read_bytes()).hexdigest()
    if not marker.is_file() or marker.read_text().strip() != fingerprint:
        print("Installing dependencies; first launch needs network access.", flush=True)
        run([str(interpreter), "-m", "pip", "install", "torch", "--index-url",
             "https://download.pytorch.org/whl/cpu"], env, root)
        run([str(interpreter), "-m", "pip", "install", "-r", "requirements.txt"], env, root)
        marker.write_text(fingerprint + "\n", encoding="ascii")
    return str(interpreter)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", action="store_true", help="Build and start with Docker Compose.")
    parser.add_argument("--check", action="store_true", help="Preflight only; no install, model download or LLM call.")
    parser.add_argument("--no-install", action="store_true", help="Use the current Python environment.")
    parser.add_argument("--offline", action="store_true", help="Use cached HF models; generation still calls the LLM API.")
    parser.add_argument("--host", help="Bind hostname/IP; defaults to loopback on the host.")
    parser.add_argument("--port", type=int, help="UI port on the host (default 7860).")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 10):
        parser.error("Python 3.10+ is required.")
    try:
        env_file = ROOT / ".env"
        required = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
        if not env_file.is_file() and (args.docker or not all(os.environ.get(k) for k in required)):
            if not args.check:
                with env_file.open("x", encoding="utf-8") as handle:
                    handle.write((ROOT / ".env.example").read_text(encoding="utf-8-sig"))
                print("Created .env from template; fill in your own LLM settings and run again.")
            else:
                print("Missing .env; create it from .env.example and configure your own LLM.")
            return 2
        env = runtime_env(ROOT, os.environ, args)
        errors = validate_inputs(ROOT, env, args.docker)
        if errors:
            for error in errors:
                print("[preflight] " + error, file=sys.stderr)
            return 2
        print("Input preflight: PASS. No credential values are displayed.", flush=True)
        if args.docker:
            if not shutil.which("docker"):
                raise RuntimeError("Docker is not installed. Install and start Docker Desktop / Docker Engine.")
            run(["docker", "compose", "version"], env)
            run(["docker", "compose", "config", "--quiet"], env)
            run(["docker", "info", "--format", "{{.ServerVersion}}"], env)
            if not args.check:
                run(["docker", "compose", "up", "--build", "--detach", "--wait", "--wait-timeout", "180"], env)
        elif args.check:
            if args.no_install:
                local_python(ROOT, env, True)
            print("Local preflight: PASS. No dependencies installed or API/model requests made.")
        else:
            interpreter = local_python(ROOT, env, args.no_install)
            print("Starting Gradio. First question builds the index; API calls may incur charges.", flush=True)
            run([interpreter, "-m", "demo.app"], env)
        if args.docker and not args.check:
            host = env.get("RAG_BIND_ADDRESS", "127.0.0.1")
            if host in {"0.0.0.0", "::"}:
                host = "127.0.0.1"
            if ":" in host:
                host = f"[{host}]"
            print(f"Gradio: http://{host}:{env.get('RAG_PORT', '7860')}")
            print("Stop: docker compose down (named model/results volumes are retained).")
        return 0
    except KeyboardInterrupt:
        print("Stopped.")
        return 130
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        # CalledProcessError omits environment values; never dump the loaded .env.
        if isinstance(exc, subprocess.CalledProcessError):
            print(f"Startup step failed (exit {exc.returncode}). Check prerequisites above.", file=sys.stderr)
        else:
            print(f"Startup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
