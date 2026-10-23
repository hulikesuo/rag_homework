#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ -n "${RAG_PYTHON:-}" ]]; then
    PYTHON="$RAG_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo 'Python 3.10+ is required. Install Python or set RAG_PYTHON.' >&2
    exit 1
fi
exec "$PYTHON" "$ROOT/scripts/start_app.py" "$@"
