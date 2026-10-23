"""全局配置：路径、模型、默认参数。

单一配置源：所有模块只 import 这里，不要在各自模块里写死路径。
"""
import os

# 国内网络下载 HF 模型默认走镜像，避免直连卡死。
# 你自己的网络直连 HF 很快的话，设 HF_ENDPOINT 为空或注释掉即可。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# 模型已缓存后，离线加载可避免每次启动都去 Hub 探测（网络不稳时会卡 30s+）。
# 首次下载模型前请勿设置；下载完成后设 HF_HUB_OFFLINE=1 即可。
if os.environ.get("HF_HUB_OFFLINE") is None and os.environ.get("RAG_OFFLINE") == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 自动加载项目根目录 .env（若有），避免每次手动 export。
_ENV_FILE = os.path.join(PROJECT_ROOT, ".env")
if os.path.isfile(_ENV_FILE):
    with open(_ENV_FILE, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ------- 路径 -------
CORPUS_DIR = os.environ.get("RAG_CORPUS_DIR", os.path.join(PROJECT_ROOT, "data", "corpus"))
EVAL_JSONL = os.environ.get("RAG_EVAL_JSONL", os.path.join(
    PROJECT_ROOT, "data", "eval", "qa_set.jsonl"))
RESULTS_DIR = os.environ.get("RAG_RESULTS_DIR", os.path.join(PROJECT_ROOT, "results"))

# 语料分块缓存：跑实验时避免反复解析原文档
INDEX_CACHE = os.path.join(RESULTS_DIR, "_cache")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(INDEX_CACHE, exist_ok=True)

# ------- 检索默认参数 -------
DEFAULT_TOP_K = 5

# 分块默认值
DEFAULT_CHUNKER = "fixed"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 128

# 混合检索融合权重（score = alpha * dense + (1 - alpha) * bm25），T4 里扫场
DEFAULT_FUSION_ALPHA = 0.5

# ------- 模型 -------
# 本地向量模型：bge-small-zh-v1.5（CPU 可跑）
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 云端 LLM API：从环境变量取，不外硬编码 key
# 用 env 覆盖，README 里说明如何填
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")

# 重排模型（cross-encoder，可选，本地/云端）
RERANK_MODEL_NAME = os.environ.get("RERANK_MODEL_NAME", "")

# ------- 评测 -------
# 知晓用 DevSet 的 LLM judge（可选）
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", LLM_MODEL)


def require_llm():
    """没有配置 LLM 时给出清晰报错，而不是中途莫名失败。"""
    if not (LLM_BASE_URL and LLM_API_KEY and LLM_MODEL):
        raise RuntimeError(
            "未配置 LLM。请设置环境变量：\n"
            "  export LLM_BASE_URL=...\n"
            "  export LLM_API_KEY=...\n"
            "  export LLM_MODEL=..."
        )
