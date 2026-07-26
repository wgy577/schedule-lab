# Causal Schedule Lab 新窗口交接文档

> **所属范围**：A–H 全局交接  
> **文档职责**：让没有当前对话历史的新 Codex/LLM 窗口快速恢复项目上下文、定位真实
> Artifact、遵守已确定边界，并从第二阶段正确继续。  
> **交接基线**：`0.7.4`  
> **交接日期**：2026-07-26

## 1. 项目位置与当前状态

项目根目录：

```text
/Users/guangyuwu/Desktop/causal_schedule_lab
```

当前目录不是 Git 仓库。不要假设已有 commit/branch，也不要擅自上传。

第一阶段“项目语义理解 + 次级目标高召回 + 题库外提案”已经暂时收口。第二阶段尚未
开始真实批量排程干预、Factor 验证或 Agentic RL 训练。

## 2. 不可误解的项目目标

1. 主要目标是优化已有甘特图/排程，而不是优先优化目标项目代码；
2. 优先复用目标项目自己的求解器、环境、Validator 和 Oracle；
3. LLM/Agent负责诊断、Factor假设、算子和预算选择；程序/求解器生成排程；Oracle裁决；
4. 代码一致性与Validator问题是安全门，不是主要搜索目标；
5. 第一阶段输出的是次级目标候选，不是最终影响因子；
6. 任何目录内或题库外候选都不能未经候选变化、干预和Oracle验证直接进入优化器；
7. 最终只接受完整可行且正式目标严格改善的候选，否则恢复 incumbent；
8. 项目目标是通用支持 JSP、FSP、FJSP、HFSP，不得把FJSP pilot写死成通用结论。

## 3. 新窗口阅读顺序

按顺序阅读，不要先从零扫描全部源码：

1. `README.md`：项目定位和总入口；
2. `docs/cross_module/STEP1_COMPLETION_AND_AGENTIC_SCHEDULING_STEP2.md`：第一阶段产物
   和第二阶段完整方案；
3. `PROJECT_STATUS_AND_ROADMAP.md`：真实状态和未完成项；
4. `PROJECT_MODULE_GRAPH.md`：三层模块与数据流；
5. `PROJECT_MODULE_GRAPH_INTERACTIVE.html`：可缩放架构图；
6. `docs/modules/module_b_semantics_memory/LLM_SEMANTIC_COMPILER.md`：LLM Harness；
7. `docs/modules/module_b_semantics_memory/SCHEDULING_SEMANTIC_KNOWLEDGE_AND_IMPACT.md`：
   题库、高召回、置信度和开放提案门；
8. `docs/modules/module_d_diagnosis_causality/CAUSAL_MODULE_WORKPLAN.md`：Factor/CIP边界；
9. `docs/modules/module_h_experiments_statistics/EXPERIMENT_PROTOCOL.md`：配对实验和统计；
10. `EXPERIMENTS.md`：已执行运行，不要把计划当结果。

## 4. 当前权威 Artifact

| Artifact | 用途 |
|---|---|
| `outputs/fjsp_test/fjsp_opus5_staged_semantics.json` | FJSP项目完整语义、证据、6约束、4决策、3 Oracle |
| `outputs/fjsp_test/fjsp_opus5_open_world_qualified_final.json` | 当前正式 Opus：14目录指标、2新指标、4固定诊断、3新诊断 |
| `outputs/fjsp_test/fjsp_mimo_open_world_qualified_final.json` | 便宜模型同Schema对照 |
| `outputs/fjsp_test/FJSP_METRIC_RECALL_COMPARISON_2026-07-26.md` | 截至当前全部FJSP相关调用/失败/修复/对照汇总 |
| `outputs/fjsp_test/*.events.jsonl` | Provider、模型、时间、Token、成本和错误Trace |

当前 Opus 新次级目标：

- `non-delay 掩码剪除对数量累积`，medium，优先进入第二阶段；
- `下界 shaping 增量总量`，low，先检查望远镜求和/公式等价，不得直接采用。

## 5. 第一阶段关键实现接口

```text
src/causal_schedule_lab/semantic_agent.py
    Navigator、三批Analyst、受控复读、短期记忆

src/causal_schedule_lab/llm_semantics.py
    语义Schema、证据引用、Project Semantics

src/causal_schedule_lab/semantic_knowledge.py
    问题族/变体/工程模式条件检索

src/causal_schedule_lab/secondary_metric_knowledge.py
    85项目录、确定性多视图召回、最多20项候选包

src/causal_schedule_lab/constraint_impact.py
    高召回Critic、置信度、schema extensions、hard validation覆盖、
    novel metric/diagnostic双通道、证据/去重/候选变化门

src/causal_schedule_lab/providers/claude_code_cli.py
    Opus-5 Claude Code CLI路由；--effort已真实透传
```

## 6. 第二阶段已有可复用接口

```text
graph.py                    build_scheduling_graph
cip.py                      CausalCoreDiscoverer / CIPRanker / ClosurePredictor
mechanisms.py               measure_mechanisms / qualify_factor / posterior estimate
operators.py                ActionIndex / build_intervention
search.py                   DeterministicNeighborhoodPortfolio / TabuMemory
repair.py                   GenericCPSATRepairGenerator
conditional_generator.py    SolverBackedConditionalGenerator
agentic_rl.py               HierarchicalActionSpace / decode_action / rewards
agent.py                    MaskedActorCritic / MaskedPPOAgent
core_validation.py          validate_schedule
validation.py               MultiFidelityValidator
controller.py               AgenticImprovementController
posterior.py                MultiFidelityPosterior
experiment_runner.py        run_methods / save_results
statistics.py               paired comparison / bootstrap / Wilcoxon/Friedman/Holm
```

代码存在不等于已训练或已接入。尤其是 Agentic RL、CIP GNN、条件生成模型和机制后验
都没有正式训练数据/checkpoint，必须保持真实状态描述。

## 7. 第二阶段第一批工作

严格按以下顺序继续：

1. 冻结第一阶段 Artifact 和知识版本，不继续无目的扩题库；
2. 为 high/medium 甘特图指标建立项目 Adapter/Calculator；
3. 读取目标项目实例、incumbent 和现有求解器接口；
4. 输出可追踪甘特图诊断：关键资源、关键块、等待传播、负荷与尾端瓶颈；
5. 从异常次级目标发现绑定具体IR实体的 Factor/Interaction；
6. 为 Factor 绑定合法 operator、closure 和 solver budget；
7. 用项目求解器或 bounded CP-SAT 生成完整候选；
8. 运行静态/项目 Oracle，记录成功和失败；
9. 做配对干预并更新后验；
10. 先让Agent处于shadow mode，再做排序/BC，最后才考虑masked PPO。

第一个建议 pilot 见
`docs/cross_module/STEP1_COMPLETION_AND_AGENTIC_SCHEDULING_STEP2.md` 第11节。

## 8. 当前必须保留的科学边界

- `confidence` 是LLM相关性软权重，不是效应大小；
- `selected` 或 `proposed` 不等于 Factor；
- 图上可达不等于因果；
- 候选间变化不等于可控；
- 单次改善不等于稳定效应；
- 局部搜索改善不等于全局最优；
- 一个FJSP案例不证明跨问题族泛化；
- 诊断失败可以阻止候选验收，但诊断项不是makespan优化奖励。

## 9. 当前已知技术债与风险

1. `下界 shaping 增量总量`可能与终态下界数学重复；
2. 题库外提案只有精确名称去重，尚无公式语义等价/自动化简；
3. 36项诊断目录尚未使用与指标相同的动态召回链；
4. 开放提案尚未自动落入长期图谱的人工审批工作流；
5. FJSP目标项目缺独立排程Validator；
6. CP-SAT参照未可靠记录求解状态；
7. Opus中转长请求偶发524/超时；失败不得无限自动重试；
8. Claude Code CLI报告的输入Token可能是中转统计边界，成本与输出Token可记录但需
   谨慎解释；
9. 当前项目目录不是Git仓库，修改前后无法依赖git diff恢复。

## 10. 安全与密钥

- `.env`含API密钥，不得打印、复制进文档、提交或上传；
- Provider Trace不保存Prompt、响应正文或密钥；
- 正式顺序为：本地测试 → 便宜模型完整链 → Opus最小探针 → Opus正式复核；
- Opus调用设置美元预算上限，不做无限重试；
- 未经用户明确授权，不上传Git、不发布Artifact、不调用外部业务系统。

## 11. 验证命令

```bash
cd /Users/guangyuwu/Desktop/causal_schedule_lab

PYTHONPATH=src .venv/bin/python -m compileall -q src tests
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python \
  .agents/skills/maintain-causal-schedule-lab/scripts/audit_framework.py --json
```

当前基线：68项测试通过；框架审计0错误、0警告；`0.7.4` wheel构建成功。

## 12. 文档维护规则

本项目任何架构、状态、公式、Agent、Oracle或实验变化都必须使用：

`.agents/skills/maintain-causal-schedule-lab/SKILL.md`

至少同步：

- `PROJECT_STATUS_AND_ROADMAP.md`；
- `PROJECT_MODULE_GRAPH.md` 与交互HTML；
- 对应A–H模块文档；
- 已执行实验进入 `EXPERIMENTS.md`；
- 测试使用 `tests/test_registry.json` 中的既有资产，避免重复写临时test脚本。

## 13. 可直接复制给新窗口的启动指令

```text
请接手 /Users/guangyuwu/Desktop/causal_schedule_lab。
先完整阅读 .agents/skills/maintain-causal-schedule-lab/SKILL.md，随后按
docs/cross_module/NEXT_WINDOW_HANDOFF.md 的阅读顺序恢复上下文。

第一阶段项目语义、高召回次级目标和开放世界提案已经暂时封板。下一步的主要目标是
优化甘特图/排程，而不是优先重构目标项目代码。请先执行第二阶段 S2.0–S2.1：冻结
第一阶段Artifact，盘点目标项目实例/incumbent/求解器接口，为high/medium排程指标
建立可追踪Calculator，并设计第一个Factor×Operator×Closure配对实验。

Agent/LLM只负责诊断和动作选择，排程由求解器生成，Oracle负责最终裁决。不要把
SecondaryTarget当成Factor，不要声称Agentic RL已训练，不要上传或泄露.env。
```
