FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HOME=/app/.cache/huggingface \
    RAG_HOST=0.0.0.0 \
    RAG_PORT=7860

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app

COPY requirements.txt ./
# CPU wheels avoid bundling CUDA libraries in a CPU-only demo image.
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app src/ ./src/
COPY --chown=app:app demo/ ./demo/
RUN mkdir -p /app/data/corpus /app/results /app/.cache/huggingface \
    && chown -R app:app /app/data /app/results /app/.cache

USER app
EXPOSE 7860
# UI liveness only: this does not download a model or call the LLM API.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('RAG_PORT', '7860') + '/config', timeout=3).close()"
CMD ["python", "-m", "demo.app"]
