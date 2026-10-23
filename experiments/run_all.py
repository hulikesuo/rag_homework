"""分发实验，默认仅运行 T2/T4/T5/T7 的离线部分。

示例：python -m experiments.run_all t2 t4 --eval data/eval/qa_set.jsonl \
    --results results/isolated
生成 API 必须显式启用 --generate；启用后也只运行所选任务，不代表完整生成实验。
"""
import argparse
import importlib
import inspect
import os


TASKS = {
    "t1": "experiments.t1_baseline",
    "t2": "experiments.t2_chunking",
    "t3": "experiments.t3_query_opt",
    "t4": "experiments.t4_retriever_compare",
    "t5": "experiments.t5_rerank_compress",
    "t6": "experiments.t6_advanced",
    "t7": "experiments.t7_eval",
}
DEFAULT_TASKS = ("t2", "t4", "t5", "t7")


def _generation_kwargs(key, run, generate):
    """不同版本仅传声明的开关；缺少必要开关时拒绝隐式执行。"""
    required = {
        "t2": "do_generate", "t3": "do_llm", "t5": "do_generate",
        "t6": "use_llm", "t7": "do_generate",
    }
    if key not in required:
        return {}
    parameters = inspect.signature(run).parameters

    def accepts(name):
        parameter = parameters.get(name)
        return parameter is not None and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)

    flag = required[key]
    if not accepts(flag):
        return None
    kwargs = {flag: generate}
    if key == "t3" and accepts("do_generate"):
        kwargs["do_generate"] = generate
    return kwargs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="运行所选实验；默认 t2/t4/t5/t7 离线，不包含完整生成实验。")
    parser.add_argument("tasks", nargs="*", metavar="TASK",
                        help="可选 t1 t2 t3 t4 t5 t6 t7；默认 t2 t4 t5 t7")
    parser.add_argument("--eval", dest="eval_jsonl", metavar="JSONL",
                        help="指定评测 JSONL；在导入任何实验前设置 RAG_EVAL_JSONL")
    parser.add_argument("--results", metavar="DIR",
                        help="指定隔离结果目录；在导入任何实验前设置 RAG_RESULTS_DIR")
    parser.add_argument("--generate", action="store_true",
                        help="显式允许所选实验调用生成 API（包括改写、LLM 重排等）")
    args = parser.parse_args(argv)
    which = args.tasks or DEFAULT_TASKS
    unknown = [key for key in which if key not in TASKS]
    if unknown:
        parser.error(f"无此任务：{', '.join(unknown)}（可选 {' '.join(TASKS)}）")

    # 本模块只导入标准库；这些值必须在 common / src.config 首次导入前就位。
    if args.eval_jsonl:
        path = os.path.abspath(os.path.expanduser(args.eval_jsonl))
        if not os.path.isfile(path):
            parser.error("--eval 必须指向存在的评测 JSONL 文件")
        os.environ["RAG_EVAL_JSONL"] = path
    if args.results:
        path = os.path.abspath(os.path.expanduser(args.results))
        if os.path.exists(path) and not os.path.isdir(path):
            parser.error("--results 必须指向目录，而非文件")
        os.environ["RAG_RESULTS_DIR"] = path
    if not args.generate:
        # 离线模式只使用本地模型缓存；不允许自动下载缺失模型。
        os.environ["RAG_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    if args.generate:
        print("已显式启用生成 API：仅运行所选任务；各任务仍可能跳过未满足条件的部分。")
    else:
        print("离线模式：生成 API 未授权，仅测所选任务的检索/离线部分；"
              "不代表完成生成实验。")
    statuses = []
    for key in which:
        if key == "t1" and not args.generate:
            reason = "T1 必须调用生成 API；未传 --generate，已跳过"
            print(f"[{key}] status=skipped：{reason}")
            statuses.append({"task": key, "status": "skipped", "reason": reason})
            continue
        module = importlib.import_module(TASKS[key])
        kwargs = _generation_kwargs(key, module.run, args.generate)
        if kwargs is None:
            reason = "当前 run 接口缺少显式生成开关，请更新对应实验后重试"
            print(f"[{key}] status=skipped：{reason}")
            statuses.append({"task": key, "status": "skipped", "reason": reason})
            continue
        print(f"\n===== 运行 {key} =====")
        module.run(**kwargs)
        mode = "generation_enabled" if args.generate else "offline"
        print(f"[{key}] status=completed mode={mode}（仅指所选任务本次运行结束）")
        statuses.append({"task": key, "status": "completed", "mode": mode})
    return statuses


if __name__ == "__main__":
    main()
