# 文档目录

项目的两个最高优先级维护入口位于仓库根目录：

- [项目状态与持续路线图](../PROJECT_STATUS_AND_ROADMAP.md)
- [两层项目模块图](../PROJECT_MODULE_GRAPH.md)
- [可缩放项目模块图](../PROJECT_MODULE_GRAPH_INTERACTIVE.html)

`docs/` 下按职责分为四类：

## architecture

系统结构、平台计划、因果模块、长期记忆、公式状态和需求追踪：

- [系统详细架构](architecture/SYSTEM_ARCHITECTURE.md)
- [Agent 平台壳计划](architecture/AGENT_PLATFORM_WORKPLAN.md)
- [因果模块实施计划](architecture/CAUSAL_MODULE_WORKPLAN.md)
- [长期记忆与因果机制](architecture/LONG_TERM_MEMORY_AND_CAUSAL_MECHANISMS.md)
- [数学公式实现矩阵](architecture/FORMULA_IMPLEMENTATION_MATRIX.md)
- [原框架追踪矩阵](architecture/TRACEABILITY.md)
- [框架需求基线](architecture/FRAMEWORK_REQUIREMENTS.md)

## semantics

LLM 项目理解、知识检索、盲测、模型比较与人工核验：

- [LLM 语义编译器](semantics/LLM_SEMANTIC_COMPILER.md)
- [无论文/碎片文档语义学习](semantics/CODE_ONLY_SEMANTIC_LEARNING.md)
- [LLM 语义盲测 Harness](semantics/LLM_SEMANTIC_HARNESS.md)
- [L2D 多模型比较](semantics/L2D_MODEL_COMPARISON_REPORT.md)
- [L2D 论文—代码人工核验](semantics/L2D_PAPER_CODE_HUMAN_VERIFICATION.md)
- [调度语义知识与影响权重](semantics/SCHEDULING_SEMANTIC_KNOWLEDGE_AND_IMPACT.md)

## experiments

实验设计、数据划分、对照、统计和复现规则：

- [实验协议](experiments/EXPERIMENT_PROTOCOL.md)

已经执行的实验结果仍统一记录在根目录 [EXPERIMENTS.md](../EXPERIMENTS.md)。

## guides

面向使用者和新项目接入的操作说明：

- [项目适配指南](guides/ADAPTER_GUIDE.md)

## 维护约束

1. 当前状态与未来任务优先更新根目录 `PROJECT_STATUS_AND_ROADMAP.md`；
2. 大模块、内部实现或数据流变化更新根目录 `PROJECT_MODULE_GRAPH.md`；
3. 专题细节再进入对应一级分类目录；
4. 已执行实验只写入 `EXPERIMENTS.md`，规划不能冒充结果；
5. 文件移动或重命名后必须运行项目文档审计和链接检查。
6. 模块、状态、数据流或实现细节变化时，Markdown 图、交互图和仍在使用的其他
   项目图片必须在同一轮修改中同步更新。
