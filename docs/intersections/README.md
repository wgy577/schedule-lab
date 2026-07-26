# 跨模块交叉索引

> **所属范围**：A–H 跨模块文档治理  
> **交叉分类**：全局索引  
> **文档职责**：按稳定的模块交叉关系聚合既有文档，不复制正文。  
> **维护触发**：文档的关联模块、数据流或职责发生变化时同步更新。

这里回答“某项能力同时经过哪些模块”。每篇正文仍只保留一个主归档目录；本目录
只提供交叉视图，避免同一份内容复制后失去同步。

| 交叉 | 主题 | 入口 |
|---|---|---|
| A × B | 项目证据 → 语义理解 | [证据与语义](a_b_evidence_semantics/README.md) |
| A × B × H | 语义提取 → 盲测与成本评估 | [语义评估](a_b_h_semantic_evaluation/README.md) |
| B × D | 语义知识 → 因果诊断 | [语义与因果](b_d_semantics_causality/README.md) |
| B × H | LLM 运行 → Token、成本与错误审计 | [LLM 运行审计](b_h_llm_operations/README.md) |
| B × D × H | 文献语义 → 指标发现 → 证据验收 | [指标发现与验收](b_d_h_metric_discovery/README.md) |
| D × E × F × H | 诊断 → 候选 → Oracle → 实验 | [优化闭环](d_e_f_h_optimization_loop/README.md) |
| A–H | 全局架构、公式与追踪 | [全局治理](a_to_h_global/README.md) |
