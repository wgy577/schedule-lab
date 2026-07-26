# 可复用测试目录

> **所属范围**：A–H 测试交叉层  
> **职责**：按模块、能力、层级和运行成本复用现有测试，避免重复新建测试文件。

测试标签的唯一注册表是 [`test_registry.json`](test_registry.json)。`conftest.py` 在
收集阶段自动应用 Pytest marker，并拒绝未注册的新测试文件或失效记录。

## 当前资产索引

| 测试文件 | 模块 | 主要能力 | 层级 / 成本 |
|---|---|---|---|
| `test_claude_code_provider.py` | B | Claude Code Provider、Skill 路由 | unit / low |
| `test_core_ir_and_solvers.py` | C × F | 统一 IR、dispatch、CP-SAT、硬约束 | integration / medium |
| `test_end_to_end.py` | C × E × F × G | Controller、端到端、回退 | system / medium |
| `test_learning_and_posterior.py` | E × H | 多任务学习、后验 | unit / medium |
| `test_llm_provider.py` | B | Provider、推理协议、重试、模型路由 | unit / low |
| `test_llm_semantics.py` | A × B | evidence packet、schema、repair | integration / low |
| `test_llm_tracing.py` | B × H | Trace、Token、错误审计 | unit / low |
| `test_memory_and_mechanisms.py` | B × D × H | 长期记忆、机制指标、后验 | integration / low |
| `test_research_pipeline.py` | D × E × F × H | 候选、Action Mask、统计 | integration / medium |
| `test_semantic_agent.py` | A × B | Navigator、读门、分阶段语义 | integration / low |
| `test_semantic_compiler.py` | A × B × C | 语义编译、证据门 | unit / low |
| `test_semantic_graph_and_taxonomy.py` | B × D | 语义图、分类树、Variant Head | integration / low |
| `test_semantic_harness.py` | A × B × H | 盲测、泄漏控制、Artifact 过滤 | integration / low |
| `test_semantic_knowledge_and_impact.py` | B × D | 知识检索、约束影响、次级指标 | integration / low |

常用调用：

```bash
# Module B 全部测试
PYTHONPATH=src .venv/bin/python -m pytest -m module_b -q

# 只测次级指标目录
PYTHONPATH=src .venv/bin/python -m pytest -m cap_secondary_metric_catalog -q

# 低成本测试
PYTHONPATH=src .venv/bin/python -m pytest -m cost_low -q

# Module B 与 Module D 的交集
PYTHONPATH=src .venv/bin/python -m pytest -m "module_b and module_d" -q
```

新增需求时按以下顺序处理：

1. 在注册表按 `cap_*` 查找已有测试；
2. 能覆盖时直接运行；
3. 缺少一个边界条件时，在已有测试文件内增加最小 case；
4. 只有出现新的独立能力边界时才新建 `test_*.py`，并同步注册标签和文件头。
