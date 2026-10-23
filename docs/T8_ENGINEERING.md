# T8-4 工程化部署

## 交付范围

- `Dockerfile`：Python 3.11 CPU 镜像、非 root 运行、Gradio UI 健康检查。
- `compose.yaml`：默认仅开放本机7860端口；语料只读挂载，模型和运行结果使用命名卷。
- `start.ps1` / `start.sh`：Windows / Bash 一键预检与启动，支持本机和 Docker 模式。
- `demo/app.py`：Gradio 问答界面，支持 hybrid/dense/bm25、top-k、上下文压缩，展示答案、证据与耗时。
- `scripts/check_engineering.py`：不下载模型、不调用 LLM 的部署回归检查。

课程 T8-4 原文还要求 **3分钟录像演示**。本实现不把脚本或截图冒充录像，也不把静态检查当作 Docker 构建通过；录制清单见末节。

## 前置准备

1. 本机安装 Python 3.10+；Docker 模式还需安装并启动 Docker Desktop / Docker Engine，并支持 Compose v2 的 `up --wait`。Windows 使用 Linux containers。
2. 自行准备获准使用的文档，放入 `data/corpus/`。支持 docx/txt/md/html；PDF 需额外安装 pypdf，默认镜像未包含它。
3. 首次启动如果没有 `.env`，脚本只创建配置模板并退出，不会编造 API 配置。填写 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL` 后重跑。已有 `.env` 不会被覆盖；进程环境变量优先。
4. 首次安装依赖、构建镜像及下载向量模型需要网络与磁盘空间；生成回答会调用自己配置的 API，可能计费。

真实 `.env`、数据、题集、实验结果、报告、压缩包与模型缓存均不提交 Git。`.dockerignore` 使用允许清单，只有 Dockerfile、依赖及 `src/`、`demo/` 的 Python 代码进入构建上下文；不在构建阶段复制配置或下载语料/模型。

## Windows 一键启动

在项目根目录执行：

```powershell
# 首次启动：自动建 .venv 并安装 CPU 依赖；后续清单未变则复用
powershell -ExecutionPolicy Bypass -File .\start.ps1

# 使用已配好的 Python 环境，不再安装依赖
$env:RAG_PYTHON = 'C:\path\to\python.exe'
.\start.ps1 -NoInstall

# 只预检：不安装、不下载模型、不调用 API、不启动服务
.\start.ps1 -NoInstall -Check

# Docker 构建并后台启动，等待 UI 健康检查
.\start.ps1 -Docker

# 可选：改端口 / 在模型已缓存后离线加载 HF 模型
.\start.ps1 -NoInstall -Port 7861 -Offline
```

浏览器访问 `http://127.0.0.1:7860`（改端口时相应调整）。PowerShell 的 `-ExecutionPolicy Bypass` 仅用于本次脚本进程，不修改系统执行策略。

## Linux / macOS / WSL / Git Bash

```bash
bash start.sh
bash start.sh --no-install --check
bash start.sh --docker
bash start.sh --no-install --offline --port 7861
# 已有环境时可显式选解释器
RAG_PYTHON=/path/to/python bash start.sh --no-install
```

`--offline` / `-Offline` 只限制 Hugging Face 模型加载，**不会禁止回答所需的远程 LLM API**。本机缓存和 Docker 命名卷是两份独立缓存：本机已缓存不代表容器已缓存，容器首次使用不能开启离线加载。

## Docker 运维

```bash
docker compose ps
docker compose logs --tail 100 rag
docker compose down
```

本机模式前台运行，Ctrl+C 停止；Docker 模式后台运行，用 `docker compose down` 停止并移除容器，但保留模型与运行结果命名卷。不要随意增加 `--volumes`，它会删除命名卷中的缓存和结果。

容器内 `RAG_HOST=0.0.0.0` 用于端口映射；宿主默认仍绑定127.0.0.1，不对局域网开放。若显式指定 `-BindAddress 0.0.0.0` / `--host 0.0.0.0`，需自己配置防火墙和认证，**不要直接公开无认证界面**。

LLM 服务在宿主机上时，Docker 内不能用127.0.0.1访问宿主服务；Docker Desktop 中将自己的 API 地址改为 `http://host.docker.internal:端口/v1`。原生 Linux Docker Engine 需要自行配置 host-gateway 或使用容器可达地址。

健康检查访问 Gradio `/config`，只证明 UI 已启动，不证明语料可解析、模型可加载或 LLM 可调用。第一次提问才构建索引和加载向量模型，需等待；完整验收必须实际提问并核对证据。预检只检查文件、必需配置和基础依赖，不探测 API 密钥是否有效，也不会把配置值打印出来。不要分享包含密钥的 `docker compose config` 完整输出。

## 验证命令

```bash
python -I -B scripts/check_engineering.py
python -I -B scripts/check_demo_acceptance_stub_test.py
```

有可用 Docker 引擎时，还需执行 `start.ps1 -Docker` / `bash start.sh --docker`，确认构建成功、`docker compose ps` 显示 healthy，再在浏览器中真实提问。如果引擎未运行或没有完成构建，不应宣称容器运行验收通过。

### 本次验证记录（2026-10-23）

- 工程离线回归27/27、现有 Gradio 边界回归4/4通过；PowerShell 与 Git Bash 一键脚本已实际执行预检。
- Compose 配置、PowerShell/Bash 语法检查通过；真实 Gradio HTTP `/config` 与空问题队列响应通过，未调用 LLM，也未下载模型。这是 UI 冒烟检查，不替代真实文档问答。
- Docker Desktop 引擎已启动，但实际构建在拉取 `python:3.11-slim-bookworm` 元数据时失败：Docker Hub 连接超时。因此 **镜像构建与容器健康运行尚未验证通过**。
- 需先让 Docker Desktop 能访问 Docker Hub（按自己的网络配置其代理或组织批准的镜像源），再执行一键 Docker 启动；不要为绕过网络问题禁用 TLS 或把 API 密钥写入 Dockerfile。
- 3分钟录像尚未录制，完整 T8-4 验收仍需完成上述 Docker 运行检查及录像。

本次功能不会重写历史实验结果、验收证据或已打包的本地评审压缩包；源码更新与历史压缩包是不同交付版本。

## 3分钟录像清单（待录制）

| 时间 | 展示内容 |
|---|---|
| 0:00–0:35 | 展示 Dockerfile、Compose、启动脚本；说明数据/密钥不入仓库和镜像 |
| 0:35–1:00 | 已预热依赖后执行一键启动，展示容器 healthy 或本机 UI 启动日志 |
| 1:00–2:10 | 打开 Gradio，输入实际问题，展示答案、来源证据、top-k、检索模式和耗时 |
| 2:10–2:40 | 切换检索模式或压缩开关再提问，说明首次模型下载、API费用与证据边界 |
| 2:40–3:00 | 展示停止命令与离线回归结果；不录制 `.env` 内容、API密钥或私人终端历史 |
