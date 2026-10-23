"""LangChain 基线链；导入本模块不会加载 LangChain 或生成器。"""


class EmptyGenerationError(ValueError):
    """生成器返回空白文本。"""


def retrieve_and_record(pipe, retrieval_kwargs, question):
    """对问题检索并把实际命中的 ID/文本复制进结果；返回 (state, result)。

    检索失败时仅在 result 里记录异常类型，供生成阶段提前返回。生成的
    上下文在检索成功后立即复制，因此后续任一步骤失败都保留本次证据。
    """
    result = {
        "answer": "", "retrieved": [], "context_chunks": [],
        "context_texts": [], "context_recorded": False,
        "rag_status": "error", "rag_error": "",
    }
    state = {"question": question, "context": [], "result": result}
    try:
        hits = pipe.retrieve(question, **retrieval_kwargs)
        # 复制实际命中的片段，避免生成阶段修改检索器持有的字典。
        context = [dict(chunk) for chunk, _ in hits]
        context_ids = [chunk["chunk_id"] for chunk in context]
        context_texts = [chunk["text"] for chunk in context]
        result.update(
            retrieved=list(context_ids),
            context_chunks=list(context_ids),
            context_texts=list(context_texts),
            context_recorded=True,
        )
        state["context"] = context
    except Exception as exc:
        result["rag_error"] = type(exc).__name__
    return state, result


def generate_after_retrieve(state, generator):
    """基于已记录的检索结果生成答案；失败仅记录异常类型并保留证据。"""
    result = state["result"]
    if result["rag_error"]:
        return result
    try:
        answer = generator.generate_rag(state["question"], state["context"])
        if not isinstance(answer, str):
            raise TypeError("生成器未返回文本")
        if not answer.strip():
            raise EmptyGenerationError("生成器返回空白文本")
        result.update(answer=answer, rag_status="ok")
    except Exception as exc:
        # 不记录第三方异常消息，避免带入凭证或服务端点。
        result["rag_error"] = type(exc).__name__
    return result


def build_baseline_chain(pipe, retrieval_kwargs):
    """构建检索 | 生成链，invoke(question) 返回基线结果字典。

    检索参数在构建时复制，每次调用单独保存上下文和结果。
    依赖只在构建链时加载；生成器只在检索成功后加载。
    """
    from langchain_core.runnables import RunnableLambda

    retrieval_options = dict(retrieval_kwargs)

    def retrieve(question):
        state, _ = retrieve_and_record(pipe, retrieval_options, question)
        return state

    def generate(state):
        # 检索失败时直接返回已记录的错误，避免加载生成器依赖。
        if state["result"]["rag_error"]:
            return state["result"]
        from . import generator

        return generate_after_retrieve(state, generator)

    return RunnableLambda(retrieve) | RunnableLambda(generate)
