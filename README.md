# Causal Schedule Lab

面向 JSP、FSP、FJSP、HFSP 与项目约束调度的“项目条件化因果核心点发现 +
Agentic 局部改进”研究平台。

本项目不是从零随机重排，也不是让大模型直接生成一张甘特图。它从一个已验证的
稳定 incumbent 出发，先定位损失出现的位置，再追踪到可修改的责任决策与最小
因果传播闭包；Agent 只决定“在哪里改、用什么算子、释放多大闭包、何时调用高成本
Oracle”，具体排程由确定性 CP-SAT/条件生成器完成，最后由验证器裁决。

> 当前仓库是独立通用工程，不依赖任何既有业务项目或外部调度代码。

> **当前阶段（2026-07-26）**：第一阶段“项目语义理解、次级目标高召回与题库外
> 提案”已经暂时封板。下一阶段的 P0 是直接改善甘特图/排程：Agent 诊断 Factor、
> Interaction、算子和局部范围，目标项目原求解器或受控 CP-SAT 生成候选，Oracle
> 裁决。一般代码优化排在其后。详见[第一阶段收口与第二阶段实施说明](docs/cross_module/STEP1_COMPLETION_AND_AGENTIC_SCHEDULING_STEP2.md)；
> 更换对话窗口时从[新窗口交接文档](docs/cross_module/NEXT_WINDOW_HANDOFF.md)开始。

## 核心闭环

```text
项目代码/文档/测试
        │
        ▼
语义编译与证据审计 ── 未证实语义只能标记 unknown
        │
        ▼
统一 IR + incumbent ──► 异构调度图
        │                    │
        │                    ▼
        │             CIP = (D, R, P, Ω)
        │                    │
        │                    ▼
        │       多任务评分 + 因果闭包预测
        │                    │
        │                    ▼
        │       短时程 Agentic RL 决策
        │                    │
        │                    ▼
        └──── 闭包外冻结 ◄─ 条件生成 / CP-SAT 局部修复
                             │
                             ▼
                  Code → Static → Light → Full Oracle
                             │
                    严格改善才接受，否则回退
```

共同因果骨架为：

```text
上游状态 A → 局部到达 L → 等待 W → 开始 T → 时长 D → 目标 J
```

## 已实现能力

- 独立统一 IR：作业、工序、候选模式、资源容量、前置关系、选择绑定、扩展约束、
  词典序目标和可验证排程。
- 项目语义编译：旧单轮链作为回归基线保留；新链先用低成本模型分片建立代码导航，
  再由高能力模型按“项目环境、目标约束、决策 Oracle”三批分析。模型只能申请复读
  相关函数，程序按 AST 调用/import 关系、白名单和预算审批，最后重新综合并审计
  引用。火山 Coding Plan 单轮 Provider 已实测；分阶段链已离线验收、尚待正式
  在线质量对照。
- 调度语义知识检索：预存 JSP/FSP/HFSP/FJSP 的经典定义、常见变体，以及运输、
  人员、维护、缓冲、换型、能耗、动态事件和数字孪生等工程模式，让 LLM 重点发现
  项目增量；知识先验不替代代码证据。
- proposed 指标知识：已统一导入 Round 3–6 的 85 个 metric、36 个 diagnostic 和
  1056 条去重关系；可选影响 Critic 由程序先召回最多 20 项，再让 LLM 做高召回
  白名单选择并保留 high/medium/low 置信层。不会把整个大题库注入 Prompt，候选也
  尚未自动进入优化器。
- 开放世界候选：若代码证据揭示题库外机制，Critic 可分别提出新次级目标和新诊断；
  新次级目标必须证明同一实例候选可变并绑定可干预决策，所有新项只进入 `proposed`
  审核池，不绕过计算器、Oracle、实验或人工晋级门。
- 长期记忆：SQLite 属性图保存经审核关系，L0–L4 FTS5 保存问题族、机制、原始证据
  和实验；外部 PDF/文本默认只进入 `proposed`，不会自动升级为事实。
- makespan 机制目标：八类确定性测量、同实例候选变化硬门、直接/验证间接控制门，
  以及按实例/状态记录的干预后验；当前独立可运行，尚未控制默认 Agent。
- 约束影响分析：默认使用 Opus 高召回结构化 Critic，分开评估可行性重要度、决策杠杆、目标敏感度
  和候选区分度；LLM 只选枚举档位，程序固定映射分数，低优化权重不能删除硬约束
  验证。DeepSeek 暂停承担候选删除、降权或最终审核，影响审核 CLI 会直接拒绝该路由；
  既有模型基准命令仍可用于历史对照。
- 四类基准适配：JSP、FSP、FJSP、HFSP，附确定性实例生成器。
- 调度异构图：工序、作业、资源、阶段及 precedence、resource sequence、
  eligibility、competition、causal edges。
- CIP 四元组：诊断点 D、责任点 R、因果路径 P、传播闭包 Ω。
- 规则召回与确定性排序基线；可替换为关系感知 GNN。
- 多任务学习头：改善量、有效性、验证成本、失败风险、排序、闭包成员、路径成员。
- 反事实与 pairwise 监督数据导出。
- 1–3 级闭包规则基线及节点级稀疏闭包学习接口。
- 层级 Agent 动作空间：`operator × closure level × control action`，带合法性 Mask。
- BC、Masked Actor-Critic 与 PPO 更新。
- 条件部分排程训练：destroy/reconstruct 样本、模式/开始偏好解码、多候选生成。
- 确定性局部 CP-SAT：incumbent hint、闭包外精确冻结、单 worker、固定 seed。
- 搜索组合：有序 VNS、平台期 bounded ALNS、禁忌记忆、多保真贝叶斯采集。
- 四级验证：代码语义、静态约束、轻量反事实代理、完整 Oracle。
- 三组因果对照：同算子随机点、同点随机算子、同闭包规模随机区域。
- 实验设施：统一 JSONL、消融矩阵、bootstrap、Wilcoxon、Cliff's delta、
  Friedman 与 Holm 校正。

详细架构与持续维护入口：

- [第一阶段收口与第二阶段 Agentic 排程优化实施说明](docs/cross_module/STEP1_COMPLETION_AND_AGENTIC_SCHEDULING_STEP2.md)
- [新窗口交接文档](docs/cross_module/NEXT_WINDOW_HANDOFF.md)
- [项目状态与持续路线图](PROJECT_STATUS_AND_ROADMAP.md)
- [三层项目模块图](PROJECT_MODULE_GRAPH.md)
- [可缩放项目模块图](PROJECT_MODULE_GRAPH_INTERACTIVE.html)
- [分类文档索引](docs/README.md)
- [跨模块交叉索引](docs/intersections/README.md)
- [可复用测试资产索引](tests/README.md)
- [系统详细架构](docs/cross_module/SYSTEM_ARCHITECTURE.md)
- [Agent 平台壳实施计划](docs/modules/module_b_semantics_memory/AGENT_PLATFORM_WORKPLAN.md)
- [因果模块实施任务](docs/modules/module_d_diagnosis_causality/CAUSAL_MODULE_WORKPLAN.md)
- [长期记忆、分层检索与因果机制目标](docs/modules/module_b_semantics_memory/LONG_TERM_MEMORY_AND_CAUSAL_MECHANISMS.md)
- [数学公式实现矩阵](docs/cross_module/FORMULA_IMPLEMENTATION_MATRIX.md)
- [原框架逐节追踪](docs/cross_module/TRACEABILITY.md)
- [LLM 项目语义编译器](docs/modules/module_b_semantics_memory/LLM_SEMANTIC_COMPILER.md)
- [Claude Code 只读代码语义 Skill](docs/modules/module_b_semantics_memory/CLAUDE_CODE_SKILL_INTEGRATION.md)
- [外部代码理解 Skill 能力调研](docs/modules/module_b_semantics_memory/EXTERNAL_SKILL_PATTERN_REVIEW.md)
- [LLM 项目语义盲测 Harness](docs/modules/module_b_semantics_memory/LLM_SEMANTIC_HARNESS.md)
- [L2D 多模型对比报告](docs/modules/module_b_semantics_memory/L2D_MODEL_COMPARISON_REPORT.md)
- [L2D 论文—代码人工核验清单](docs/modules/module_a_input_evidence/L2D_PAPER_CODE_HUMAN_VERIFICATION.md)
- [调度语义知识库与优化影响权重](docs/modules/module_b_semantics_memory/SCHEDULING_SEMANTIC_KNOWLEDGE_AND_IMPACT.md)
- [次级指标与诊断项扩充研究任务书](docs/modules/module_d_diagnosis_causality/SECONDARY_METRIC_AND_DIAGNOSTIC_EXPANSION_PLAN.md)
- [无论文/碎片文档语义学习](docs/modules/module_b_semantics_memory/CODE_ONLY_SEMANTIC_LEARNING.md)
- [项目架构维护 Skill](.agents/skills/maintain-causal-schedule-lab/SKILL.md)

## 目录

```text
causal_schedule_lab/
├── configs/                    # 语义 DSL 与实验配置
├── PROJECT_STATUS_AND_ROADMAP.md # 当前状态、缺口与后续任务
├── PROJECT_MODULE_GRAPH.md     # 三层项目结构图：大模块、实现细节、真实组件映射
├── docs/
│   ├── modules/                # 按 A–H 系统模块归档的专题文档
│   │   ├── module_a_input_evidence/
│   │   ├── module_b_semantics_memory/
│   │   ├── module_c_ir_adapters/
│   │   ├── module_d_diagnosis_causality/
│   │   ├── module_e_candidate_generation/
│   │   ├── module_f_validation_oracles/
│   │   ├── module_g_acceptance_rollback/
│   │   └── module_h_experiments_statistics/
│   ├── cross_module/           # 系统总览、公式矩阵与需求追踪
│   └── intersections/          # A×B、B×D 等交叉能力索引
├── examples/manifests/         # 四类内置演示
├── scripts/reproduce_all.sh    # 一键安装、测试和四类 smoke run
├── src/causal_schedule_lab/
│   ├── ir.py                   # 独立统一调度 IR
│   ├── semantic_compiler.py    # 项目语义编译与证据门
│   ├── llm_semantics.py        # LLM 选择题语义解析与严格 JSON
│   ├── semantic_agent.py       # 分片导航、分批分析、受控复读与短期记忆
│   ├── semantic_knowledge.py   # JSP/FSP/HFSP/FJSP 知识检索
│   ├── secondary_metric_knowledge.py # 85 项统一目录、确定性多视图召回与 Token 有界候选包
│   ├── constraint_impact.py    # 情境化约束优化影响 Critic
│   ├── knowledge/              # 问题族、机制目标与 proposed 多视图指标知识
│   ├── storage/                # SQLite 图谱、证据索引与干预记忆
│   ├── mechanisms.py           # 机制计算、资格门与效应后验
│   ├── semantic_harness.py     # 论文隐藏标签与代码盲测
│   ├── providers/              # 火山/OpenAI 兼容 Provider
│   ├── graph.py                # 异构调度图
│   ├── cip.py                  # CIP 召回、责任路径与闭包
│   ├── learning.py             # 关系 GNN 与多任务头
│   ├── agentic_rl.py           # 完整层级动作空间
│   ├── agent.py                # BC / Masked PPO 基线
│   ├── conditional_generator.py# 部分排程训练与条件修复
│   ├── search.py               # VNS / ALNS / tabu / acquisition
│   ├── posterior.py            # 多保真后验
│   ├── core_validation.py      # 通用硬约束
│   ├── validation.py           # 四级 Oracle
│   ├── experiment_runner.py    # 基准与消融执行
│   └── statistics.py           # 统计检验
└── tests/                      # 已登记、可按模块/能力复用的测试资产
    ├── README.md
    └── test_registry.json      # 测试标签唯一事实来源
```

## 安装

需要 Python 3.11+。

```bash
cd /Users/guangyuwu/Desktop/causal_schedule_lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[dev]'
```

## 快速运行

运行全部测试：

```bash
.venv/bin/python -m pytest -q
```

建立当前知识图谱与分层证据库：

```bash
.venv/bin/causal-schedule-lab memory-build
.venv/bin/causal-schedule-lab memory-status
.venv/bin/causal-schedule-lab memory-search \
  --query "AGV transport synchronization" \
  --family FJSP
```

把新综述放入待审核证据层：

```bash
.venv/bin/causal-schedule-lab memory-ingest-document \
  --file /path/to/review.pdf \
  --title "Review title" \
  --source-kind peer_reviewed_survey \
  --scope FJSP
```

测量一个已接入项目当前排程的 makespan 中间机制：

```bash
.venv/bin/causal-schedule-lab measure-mechanisms \
  --project examples/manifests/jsp.json \
  --output outputs/jsp_mechanisms.json
```

审计一个项目的代码语义：

```bash
.venv/bin/causal-schedule-lab compile-semantics \
  --project-root /path/to/scheduling-project \
  --output outputs/semantic_compilation.json
```

运行四类内置实例：

```bash
.venv/bin/causal-schedule-lab demo --family jsp  --mode optimize
.venv/bin/causal-schedule-lab demo --family fsp  --mode optimize
.venv/bin/causal-schedule-lab demo --family fjsp --mode optimize
.venv/bin/causal-schedule-lab demo --family hfsp --mode optimize
```

完整可复现 smoke workflow：

```bash
bash scripts/reproduce_all.sh
```

LLM 语义编译（读取本地 `.env`，输出仍需人工审核）：

```bash
PYTHONPATH=src .venv/bin/python -m causal_schedule_lab.cli \
  compile-semantics-llm \
  --project-root . \
  --env-file .env \
  --provider-prefix SEED \
  --output outputs/llm_semantic_compilation.json
```

新的分阶段链（默认低成本 MiMo 导航、官方 Claude Code 客户端调用 Opus-5
并使用项目只读 Skill 做强分析）：

```bash
PYTHONPATH=src .venv/bin/python -m causal_schedule_lab.cli \
  compile-semantics-staged \
  --project-root . \
  --env-file .env \
  --navigator-provider-prefix MIMO \
  --analyst-protocol claude-code \
  --claude-code-project-skill \
  --max-rounds-per-batch 3 \
  --max-reads 18 \
  --output outputs/staged_semantic_compilation.json
```

建议先加 `--dry-run` 检查哪些非论文文件会进入导航；该模式不调用 API。
正式运行会实时打印每次调用的模型、阶段、批次、轮次和修复次数，并写入
`<output>.events.jsonl`。日志不保存 Prompt、响应正文或密钥，可用来判断究竟卡在
哪一个模型调用；Skill 模式还会记录实际使用过的 `Read/Glob/Grep` 名称和次数。

运行网上公开论文—官方代码盲测案例：

```bash
.venv/bin/causal-schedule-lab run-semantic-harness \
  --case examples/harness_cases/l2d.json \
  --env-file .env \
  --provider-prefix SEED \
  --max-paper-characters 36000 \
  --max-code-characters 42000 \
  --output outputs/l2d_semantic_harness.json
```

论文只生成隐藏标签，代码分析端不会收到论文文本或标签。第三方案例内容放在
被忽略的 `external_cases/`，不复制进本仓库。

使用已有固定标签比较另一个 OpenAI 兼容模型：

```bash
.venv/bin/causal-schedule-lab run-code-semantic-benchmark \
  --case examples/harness_cases/l2d.json \
  --label-artifact outputs/l2d_semantic_harness.json \
  --provider-prefix DEEPSEEK \
  --protocol openai \
  --max-code-characters 42000
```

Anthropic Messages 兼容中转使用 `--protocol anthropic`。当前五模型实测结论和
限制见 [对比报告](docs/modules/module_b_semantics_memory/L2D_MODEL_COMPARISON_REPORT.md)。

## 接入新项目

1. 将问题和 incumbent 转为 `ir.Problem` / `ir.Schedule`，或实现
   `SchedulingProjectAdapter`。
2. 对未知项目调用 LLM 语义编译 Provider 做多轮理解，再以代码、测试、文档
   证据和人工审核确认；无 LLM 时只能索引和审计已有语义。
3. 在 manifest 中声明问题族、目标、硬约束、生成器和可选领域 Oracle。
4. 对 IR 已表达的约束使用通用验证；轨迹、仿真、数字孪生等未建模约束通过
   `domain_oracle` 插件接入。
5. 先生成反事实数据和 BC 示范，再训练多任务 CIP/闭包模型与短时程策略。
6. 只有 Full Oracle 合法且词典序严格改善的候选才能更新 incumbent。

详见 [项目适配指南](docs/modules/module_c_ir_adapters/ADAPTER_GUIDE.md)。

## 可复现性与研究边界

- 固定 seed、单 worker、确定性预算、incumbent hash 与候选签名全部进入日志。
- 搜索只释放声明闭包，闭包外操作必须保持原 mode/start/end。
- 学习模型只排序和控制预算，不越过求解器与 Oracle 的硬约束。
- 没有完整 Oracle 的项目不会被宣传为“已验证可行”。
- 当前仓库提供完整研究流水线和 smoke tests；跨数据集性能结论必须在真实基准
  实验运行后，依据统计报告给出，不预先伪造结果。

## License

当前为研究原型。正式发布前请补充所需许可证与第三方数据许可说明。
