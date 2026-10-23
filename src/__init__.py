"""RAG 作业源码包。

模块划分（高内聚 / 低耦合）：
- config:      全局配置（路径、模型、默认参数），所有模块从这里取
- loader:     读取语料文档（txt/md/pdf/html）
- chunker:    各种分块策略
- embed:      向量化（sparse/dense 统一接口）
- index_store:检索索引的构建与查询（灯FAISS + kNN，可扩展）
- retriever:  检索器，组合 index_store / 打分
- rerank:     重排 + 压缩
- generator:  LLM 生成
- pipeline:   串联成一次完整检索生成（基准链路）
- utils:      公共工具（路径、计时、日志、权重保存）
"""