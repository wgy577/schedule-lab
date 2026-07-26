# 文档目录

> **所属范围**：跨模块文档治理  
> **归档规则**：每篇文档必须指定一个主模块；跨模块总览进入 `cross_module/`；交叉检索进入 `intersections/`。  
> **维护触发**：文件新增、改名、职责变化或架构节点变化时同步更新本页。

项目最高优先级维护入口：

- [项目状态与持续路线图](../PROJECT_STATUS_AND_ROADMAP.md)
- [三层项目模块图](../PROJECT_MODULE_GRAPH.md)
- [可缩放项目模块图](../PROJECT_MODULE_GRAPH_INTERACTIVE.html)

## A–H 模块目录

| 模块 | 职责 | 文档入口 |
|---|---|---|
| A | 项目输入与证据 | [Module A](modules/module_a_input_evidence/README.md) |
| B | 语义理解与知识记忆 | [Module B](modules/module_b_semantics_memory/README.md) |
| C | 统一调度表示与项目适配 | [Module C](modules/module_c_ir_adapters/README.md) |
| D | 诊断与因果机制 | [Module D](modules/module_d_diagnosis_causality/README.md) |
| E | Agent 决策与候选生成 | [Module E](modules/module_e_candidate_generation/README.md) |
| F | 求解与多保真验证 | [Module F](modules/module_f_validation_oracles/README.md) |
| G | 严格接受与回退 | [Module G](modules/module_g_acceptance_rollback/README.md) |
| H | 实验记忆与统计 | [Module H](modules/module_h_experiments_statistics/README.md) |

跨模块文档：

- [第一阶段收口与第二阶段 Agentic 排程优化实施说明](cross_module/STEP1_COMPLETION_AND_AGENTIC_SCHEDULING_STEP2.md)
- [新窗口交接文档](cross_module/NEXT_WINDOW_HANDOFF.md)
- [系统详细架构](cross_module/SYSTEM_ARCHITECTURE.md)
- [数学公式实现矩阵](cross_module/FORMULA_IMPLEMENTATION_MATRIX.md)
- [原框架追踪矩阵](cross_module/TRACEABILITY.md)

## 按模块交叉关系检索

[跨模块交叉索引](intersections/README.md)按 A × B、A × B × H、B × D、B × H、
B × D × H、D × E × F × H 和 A–H 全局关系分类。同一正文不复制，只从不同交叉
入口引用。

测试文件的模块、能力、层级和成本标签见
[`tests/README.md`](../tests/README.md)及唯一注册表
[`tests/test_registry.json`](../tests/test_registry.json)。

## 强制维护规则

1. 新文档必须放入唯一主模块目录，并在标题后标明模块、文档职责和维护触发条件；
2. 同时覆盖多个模块的总览、公式矩阵和追踪表进入 `cross_module/`；
3. 涉及两个及以上模块的正文必须声明 `交叉分类`，并加入对应的 `intersections/` 索引；
4. 状态与未来任务更新根目录 `PROJECT_STATUS_AND_ROADMAP.md`；
5. 大模块、数据流或实现节点变化同步更新两种项目图；
6. 已执行实验写入根目录 `EXPERIMENTS.md`，规划不能冒充结果；
7. 新增测试文件前先检索标签注册表并扩展既有测试；确需新建时必须登记；
8. 移动文件后必须运行 Markdown 链接检查和完整测试。
