# 项目状态与持续路线图

> 文档性质：长期维护的状态总表  
> 当前版本：`0.7.4`  
> 最近更新：2026-07-26  
> 更新责任：每次代码、知识、实验、训练、Oracle 或发布状态变化时同步维护。

## 1. 当前里程碑

| 里程碑 | 状态 | 可验证证据 | 下一门槛 |
|---|---|---|---|
| 通用 JSP/FSP/FJSP/HFSP IR | 主闭环已运行 | 四类 manifest 与端到端测试 | 更多公开实例 importer |
| LLM 项目语义编译 | 主闭环已运行 | 真实 Provider Artifact 与证据审计 | 分阶段链在线对照和人工落库 |
| 分阶段语义 Agent | 可选链已在线运行 | Navigator 分片、三批 Analyst、AST 复读门、短期记忆、完整 FJSP Artifact 和离线测试 | 用已确认标签运行跨项目在线对照 |
| 经典问题族与工程模式知识 | 主闭环已运行 | JSON seed 与检索测试 | 新综述人工审核 |
| 调度知识继承树 | 已实现并测试 | 车间调度→问题族→Head 单选 L3；无命中回退 classic；SQL 祖先合并测试 | 审核新增 L3/L4 工程变体与显著 Head |
| 项目约束语义图 | 已实现并接入 LLM 输出 | 环境、目标、约束、决策、Oracle、代码证据与 IR 关系测试 | 人工审核后的因果边与图查询 UI |
| SQLite 图谱长期记忆 | 代码已实现但未接入 | 73 节点、85 边迁移 smoke；幂等测试 | 接入语义编译默认检索 |
| L0–L4 分层证据检索 | 代码已实现但未接入 | 72 证据块、FTS5 查询和 PDF/文本导入测试 | 冲突审核 UI/CLI 与 embedding 可选重排 |
| makespan 机制目标库 | 代码已实现但未接入 | 8 个定义、8 个计算器入口 | 项目语义路由与跨族校准 |
| 候选变化与可控性资格门 | 代码已实现但未接入 | constant、hypothesized、verified 单测 | 真实 replay 生成控制证据 |
| 干预记录与效应后验 | 代码已实现但未接入 | SQLite 去重、作用域查询、保守后验单测 | Controller 自动写入与预算采集 |
| CIP/局部修复/四级验证 | 主闭环已运行 | 现有端到端测试 | 学习式 CIP 与更多领域 Oracle |
| 跨问题族效果结论 | 尚未实现 | 无真实完整实验矩阵 | 公开基准、消融与统计 |
| 可复用测试资产治理 | 已实现并测试 | 14 个测试文件已登记模块、能力、层级、成本标签；Pytest 自动 Marker 与审计门 | 新能力优先扩展既有测试资产 |
| 跨模块文档检索 | 已实现并测试 | 6 类交叉索引；正文唯一归档、交叉入口引用 | 模块职责变化时持续同步 |
| 次级指标与诊断项扩充 | 可选 Critic 运行链已接入 | 85 个 proposed metric、36 个 diagnostic、1056 条去重关系；最多 20 项高召回白名单；题库外 metric/diagnostic 双通道、代码证据、候选变化、去重和 proposed 门；当前 MIMO/Opus Artifact | 建立跨族 Recall/Precision/NDCG、开放提案归一/等价审核、计算器、Oracle 配对与人工盲测标签；不得自动进入优化器 |
| 第一阶段：项目理解与次级目标入口 | 主闭环已运行（阶段暂时封板） | FJSP 语义 Artifact、14 个目录指标、2 个题库外指标、4+3 个诊断项及完整调用 Trace | 冻结 Artifact；只有新项目出现明确缺失机制时才继续扩充 |
| 第二阶段：Factor × Operator × Closure 实验闭环 | 尚未实现 | 已有图、CIP、算子、局部修复、验证和统计接口，但没有真实批量干预数据 | 先实现甘特图指标计算器与项目求解器 Adapter，再做配对干预和 Agent shadow mode |

## 2. 当前长期记忆实物

生成命令：

```bash
causal-schedule-lab memory-build
```

默认位置：

```text
outputs/memory/schedule_memory.sqlite3
```

当前种子规模：

```text
78 nodes
123 edges
325 unique aliases
72 evidence chunks
0 real intervention experiments
```

数据库是可重建 Artifact，不作为人工编辑源。人工维护源仍是：

- `knowledge/scheduling_families.json`；
- `knowledge/mechanism_targets.json`；
- 后续审核通过的知识清单和外部证据；
- 不可变实验记录。

## 3. 您后续需要提供的信息

优先寻找当前知识种子未充分覆盖的综述、工程论文或高质量说明：

| 优先级 | 信息缺口 | 希望从资料中确认 |
|---|---|---|
| P0 | 各问题族机制与 makespan 的关系 | 哪些中间指标在什么条件下具有稳定解释力 |
| P0 | 工业约束交互 | 运输×机器、人员×机器、维护×排序、buffer×blocking |
| P0 | 常见局部算子与适用条件 | 算子直接控制什么变量、会释放多大传播闭包 |
| P1 | 动态/随机调度 | 状态变化、重调度触发、稳定性与鲁棒性 |
| P1 | 多保真验证 | 代理误差、仿真成本和 Full Oracle 选择 |
| P1 | 因果/反事实调度研究 | 干预单位、控制组、识别假设和失败边界 |
| P2 | 规模与迁移 | 跨实例/跨问题族可以共享哪些后验 |

资料可以是 PDF、Markdown 或纯文本。第一步只进入 `proposed` 证据层：

```bash
causal-schedule-lab memory-ingest-document \
  --file /path/to/review.pdf \
  --title "Review title" \
  --source-kind peer_reviewed_survey \
  --scope FJSP
```

导入不等于采信。随后需要执行：

```text
提取候选声明
→ 与现有图谱去重
→ 检查来源和冲突
→ 代码/论文/实验分层
→ 人工审核
→ active 或 rejected
```

## 4. 下一阶段实施顺序

当前优先级固定为：P0 甘特图/排程改善；P1 可行性和比较口径守门；P2 仅处理会限制
排程搜索或污染目标的代码问题；P3 论文、展示和非运行信息。完整实施包见
[`docs/cross_module/STEP1_COMPLETION_AND_AGENTIC_SCHEDULING_STEP2.md`](docs/cross_module/STEP1_COMPLETION_AND_AGENTIC_SCHEDULING_STEP2.md)。

### P0：建立甘特图 Factor—算子实验闭环

- 冻结第一阶段语义、题库和开放提案 Artifact；
- 为 high/medium 次级目标实现可追踪到资源/工序的 Calculator；
- 用 incumbent 定位关键资源、关键块、等待传播和尾端瓶颈；
- Agent 提出绑定实体的 Factor/Interaction，并选择 operator、closure 和 solver budget；
- 优先调用目标项目自己的求解器，必要时使用闭包外冻结的 bounded CP-SAT；
- 以完整可行且正式目标严格改善为唯一接受条件，记录所有失败候选；
- 先运行 shadow mode 和上下文排序器，积累 transition 后再考虑 BC/PPO。

### P1：在线验收分阶段语义链

- 用低成本 Navigator 与高能力 Analyst 分开运行；
- 比较旧单轮链和新分阶段链的约束召回、错误引用、Token 与人工修正量；
- 测试 `sufficient/partial/insufficient/contradictory` 的读请求校准；
- 暂不以单个 smoke 宣称质量提升。

### P1：把已实现记忆接入语义链

- 分阶段链已可选执行图邻域和 L0–L4 检索；下一步验证召回质量后再进入默认 runtime；
- LLM 输出中记录使用过的 node/chunk ID；
- 提案只能进入 `proposed`；
- 人工审核后才写 `active` 关系。

### P0：建立真实机制干预数据

- 在同一 instance/incumbent/state 下生成受控局部候选；
- 记录因子变化、机制变化、传播闭包和最终目标变化；
- Full Oracle 失败仍写入记忆；
- `verified_indirect` 必须来自 replay，而不是 LLM 判断。

### P1：后验进入候选排序

- 先离线 shadow mode，只输出建议不控制搜索；
- 比较无记忆、LLM 先验、经验后验三种排序；
- 校准有效率、保守收益、时间和 Token 成本；
- 通过后才允许影响 CIP/Agent 的 attention。

### P1：跨问题族验证

- JSP/FSP/FJSP/HFSP 分层实验；
- leave-one-family-out；
- 固定实例划分，禁止训练/测试泄漏；
- 报告未改善、不可行和 Oracle 失败，而不只报告最好结果。

## 5. 尚未解决的设计决策

| 决策 | 当前默认 | 需要什么证据后再修改 |
|---|---|---|
| 间接可控门槛 | 至少 1 次 replay 且一致率 ≥ 0.8 | 多实例重放稳定性 |
| 图数据库 | SQLite | 单库规模、并发和查询延迟达到迁移阈值 |
| 向量检索 | 不作为事实源；尚未接入 | 证明在同子树内提高召回且不增加错族 |
| 后验模型 | Beta 有效率 + 保守收益统计 | 足够干预样本后比较分层贝叶斯模型 |
| 三元交互 | 默认不测 | 成对模型持续出现大残差 |
| 跨族迁移 | 默认隔离 | leave-one-family-out 有统计证据 |

## 6. 每次更新必须维护的文件

| 变化 | 必须同步 |
|---|---|
| 大模块或数据流 | `PROJECT_MODULE_GRAPH.md`、`docs/cross_module/SYSTEM_ARCHITECTURE.md` |
| 模块图、流程图、截图或其他当前项目图片 | `PROJECT_MODULE_GRAPH.md`、`PROJECT_MODULE_GRAPH_INTERACTIVE.html`、对应展示入口；同步更新或明确退役 |
| 当前状态或路线 | 本文件 |
| 长期记忆/机制 | `docs/modules/module_b_semantics_memory/LONG_TERM_MEMORY_AND_CAUSAL_MECHANISMS.md` |
| 公式或统计 | `docs/cross_module/FORMULA_IMPLEMENTATION_MATRIX.md` |
| 因果数据/CIP | `docs/modules/module_d_diagnosis_causality/CAUSAL_MODULE_WORKPLAN.md` |
| 平台/存储/Provider | `docs/modules/module_b_semantics_memory/AGENT_PLATFORM_WORKPLAN.md` |
| 已执行实验 | `EXPERIMENTS.md` |
| 用户入口 | `README.md` |

## 7. 状态更新模板

```markdown
### YYYY-MM-DD / change-id

- 变化：
- 代码：
- 状态从：
- 状态到：
- 验证：
- 产物：
- 仍未完成：
- 下一步：
```

只有实际测试、运行路径和 Artifact 能推动状态升级。路线图或 Prompt 不能证明实现。

## 8. 最近维护记录

### 2026-07-26 / step1-freeze-and-agentic-scheduling-step2-plan-v1

- 阶段：第一阶段“项目语义理解、次级目标高召回和开放提案”暂时封板；没有把
  SecondaryTarget、LLM confidence 或图上可达关系误报为真实影响因素；
- 优先级：甘特图/排程改善为 P0，候选可行性与比较口径为守门，通用代码优化及其他
  非排程内容排在后面；
- 产物：新增第一阶段产物清单、第二阶段 S2.0–S2.6 实施说明和新窗口交接文档；
- 下一步：用目标项目原求解器或 bounded CP-SAT 做
  `Factor × Operator × Closure × Budget` 配对干预，先 shadow mode，再学习排序/BC，
  最后才考虑 masked PPO；
- 状态边界：Agentic RL、CIP GNN 和条件生成模型仍无正式训练数据或 checkpoint，
  不得声称已训练或已验证跨族收益。

### 2026-07-26 / complete-architecture-and-fjsp-impact-replay-v1

- 架构图：补全第一层 A–H 的真实职责与知识/Harness/插件支撑，第二层显式展开文件
  清单、AST/调用索引、三批任务、结构化读请求、程序读门、精确 symbol 证据、插件
  Oracle 和重放审计；Markdown 与可缩放 HTML 同步；
- 流程：新增当前已完成的“项目语义理解链”和“incumbent 改进链”，并单列代码阅读
  Agent 引导库的组成与权限边界；
- 工具：新增 `review-constraint-impact`，可在同一保存语义/证据上只重放指标 Critic，
  无需重新支付整条代码阅读链；同时兼容大题库接入前的旧 Artifact；
- 召回修正：变体只由结构化、带证据的 constraint kind 激活，不再把摘要中的
  “未见 setup/blocking”误识别为变体；
- 在线对照：MIMO 修正后从 85 项中得到 15 项合格、12 项 Prompt 候选并选择 3 项，
  使用 15,758 Token；Opus 在 134.6 秒遭上游 Cloudflare 524，无结果且未盲目重试；
- 状态边界：该结果只验证 Harness/召回链可运行，不晋级任何次级目标为真实影响因素。

### 2026-07-26 / deterministic-large-metric-catalog-runtime-v1

- 知识：导入 Round 5 的 12/5/171 增量和 Round 6 的 14/6/257 增量；统一为 85 个
  metric、36 个 diagnostic、1056 条去重关系；
- 兼容：原 active 11 项全部绑定到 canonical ID；四个不等价概念新增独立实体，禁止
  用近似公式强行合并；
- 召回：程序按 family、显式 variant、mechanism、modifiable decision 和 evidence
  确定性排序，缺失字段标记为 `needs_adapter_or_evidence`；
- Token：最多 12 项紧凑 JSON 进入一次 Critic 请求，不注入 85 项全文；保存字符估计、
  批次、原始合格数和上下文 fingerprint；
- 安全：LLM 只能返回召回包中的 candidate ID；重复 ID、未知 ID、未知来源约束和
  hard validation 均由程序拒绝；
- 状态边界：指标选择链已接入可选 Critic，但 36 项动态诊断选择、adapter 实际字段
  审计、人工校准和 proposed→active/优化器晋级仍未完成。

### 2026-07-26 / implemented-only-mapped-project-graph-v1

- 映射：第三层所有节点在标题中标出对应第二层 ID，并增加父子映射表；
- 范围：移除未实现的确定性召回器、MCQ、完整递归 Factor Discovery 和 Full Domain
  Oracle 节点；未来能力只保留在路线图；
- 真实性：删除不存在的 `D11 → D8/D10` 运行边，明确当前固定目标支路与 Round 4
  proposed 知识支路尚未合并；
- 排版：第一层收敛为 A→H 主链，第二层减少非必要跨区回线，第三层按父模块分栏；
  交互图改为水平/垂直正交折线。

### 2026-07-26 / three-layer-target-to-factor-boundary-v1

> 历史记录：该初版展示方式已由上方 `implemented-only-mapped-project-graph-v1`
> 取代；目标—因素发现流程继续保留在专题设计文档和路线图，不再作为当前项目图节点。

- 项目图：从两层扩展为三层；第三层专门展示“项目语义与题库→次级目标资格门→
  因素/交互发现→干预、Oracle 与后验”；
- 边界：题库和 LLM 选择题最多输出项目级次级目标，不能输出真实影响因素；Factor
  必须重新绑定代码/IR 实体并经过变化、可控性、传播、干预、Oracle 和统计门；
- 状态：Round 4 仍只完成 proposed 题库和只读接口；确定性召回、MCQ 运行链和完整
  Factor 闭环均未因此升级；
- 可视化：Markdown Mermaid 与可缩放 HTML 同步增加第三层入口。

### 2026-07-26 / round4-secondary-metric-knowledge-import-v1

- 知识物化：导入 Round 3 的 55 个 canonical candidate、25 个 diagnostic、623 条
  membership 和 44 个 gap；Round 4 将 gap 归入 9 个能力组并为全部候选建立
  semantic/computability 分离契约；
- 代码：新增 `SecondaryMetricKnowledgeBase` 只读接口，支持候选、诊断、关系、能力组、
  可计算性和参考场景查询；
- 状态：D11 从“接口已预留/计划”升级为“代码已实现但未接入”；默认语义编译、Critic
  和优化器仍不消费该知识；
- 证据边界：21 个场景保持 `design_reference_not_independent_truth`，所有 canonical
  candidate 保持 `proposed`；
- 验证：确定性重建、JSON/JSONL 解析、ID 引用与数量审计通过；复用
  `test_semantic_knowledge_and_impact.py` 完成加载、角色变化和可计算性边界测试。

### 2026-07-26 / high-recall-impact-review-and-metric-research-v1

- 模型路由：单轮影响审核默认 Anthropic/Opus；分阶段影响审核默认 Claude Code/Opus；
  OpenAI-compatible 前缀回退为 `SEED`，DeepSeek 不再是任何审核入口的默认值；
- 审核边界：DeepSeek 暂停候选删除、降权和最终审核，影响审核 CLI 直接拒绝该路由；
- 研究计划：新增次级指标与诊断项扩充任务书，按问题族综述、工程变体、原始机制
  论文和真实案例四层检索；
- 验收机制：候选经历高召回发现、公式归一、独立 Opus 复核、程序门、候选变化测试、
  Oracle 配对实验和人工批准，未验证项只进入 `proposed`。

### 2026-07-26 / reusable-tests-and-intersection-index-v1

- 测试资产：14 个既有 `test_*.py` 全部登记到 `tests/test_registry.json`，按模块、
  能力、测试层级和成本分类，并由 Pytest 自动施加 Marker；
- 复用门：未登记测试、失效记录或缺少首行 `TEST-TAGS` 会在收集/审计阶段失败；
- 文档交叉：新增 A × B、A × B × H、B × D、B × H、D × E × F × H 和 A–H
  六类交叉索引，正文仍保持唯一主模块归档；后续已扩充 B × D × H；
- 维护约束：Skill、项目审计、README 和文档索引同步加入“先复用、后扩展、最后才
  新建测试”及交叉分类规则。

### 2026-07-26 / semantic-constraint-graph-and-taxonomy-v1

- 变化：新增证据关联项目约束图，覆盖传统析取关系之外的环境、模式、资格、
  目标、决策、Oracle、未知项和代码证据；
- 知识：SQLite 采用“车间调度→问题族→变体”的单继承树，每个节点只存本层
  `local_delta`，查询时返回祖先链、合并结果和逐层贡献；
- 接入：单轮和分阶段 LLM 语义编译结果自动携带 `constraint_graph`；
- 命令：新增 `memory-taxonomy`；
- FJSP 在线验收：官方 FJSP-DRL 仓库被正确归类为 FJSP；提取 6 项约束、
  4 项决策、3 个 Oracle 和 5 个待确认问题；55 条代码证据全部通过路径/symbol
  审计；项目约束图含 83 节点、82 边；
- 影响判断：机器资格与机器排序为关键优化杠杆；恒为 0 的 release time 被降为
  固定退化参数；代码特有的 non-delay 动作掩码被识别为高杠杆实现限制，而非
  FJSP 硬约束；
- 调用审计：8 次模型调用、76,657 Token、26 次只读工具调用、0 次 schema 修复，
  Opus 调用成本约 4.84 美元（Navigator 未返回成本）；
- 验证：52 项测试通过，FJSP 证据通过率 100%。

### 2026-07-26 / fjsp-code-paper-label-audit-v1

- 分类树：禁止“各类变体/其他变体”等占位节点；新增类型必须成为独立分支，
  并允许继续扩展第四级及更深层；
- 审计原则：代码决定项目实际语义，论文只用于核对标签；
- 正确标签：标准 FJSP、单一 makespan、三项经典硬约束、工序/机器联合决策、
  所有作业时刻 0 到达、立即开工式 MDP；
- 待修正标签：移除 `multi_resource`，reward shaping 不作第二优化目标，求解算法
  选择不作排程决策；
- 风险收紧：归一化资格问题只在整个输入批次完全没有原始 0 时触发；
- 影响报告：明确降级为 `structured_llm_prior`，不是实验验证的因果权重；新增
  “约束满足”与“约束活跃度/松弛量/机制”分离规则；
- 验证：53 项测试通过。

### 2026-07-26 / agent-memory-read-unification-and-secondary-targets-v1

- 推理：Opus 5 分析与影响 Critic 默认由 `max` 降为 `high`，CLI 可单独覆盖；
- 读取：默认 `claude-code-read-mode=harness`，Claude Code 不再直接读文件，
  所有复读统一经过 AST ReadGate；保留 `claude` 模式用于单独代码审查；
- 记忆：SQL 继承树解析结果通过 `taxonomy_profiles` 注入 Agent。当时用于生成验证
  问题和变体扫描；该方式已由下一版本的单分支 Head Router 取代；
- 图：新增 `scheduling_core` 与 context/validation/evidence/audit overlay，避免审计
  信息淹没工件—工序—机器—顺序—约束主图；
- 影响：Critic schema 新增结构化 `secondary_targets`；旧 FJSP 结果已重建 8 个
  次级目标供审阅，真实因素留给递归受控实验；
- 结构图：新增 Secondary Target Discovery 与 Recursive Factor/Interaction Discovery。
- 验证：54 项测试通过；FJSP 新链 dry-run 通过。

### 2026-07-26 / variant-head-router-and-target-separation-v1

- 变体路由：不再把某问题族的全部兄弟变体注入 Agent；先由少量高显著度 Head
  确定唯一 L3 分支，未命中时回退 classic；只解析所选分支的祖先链；
- 证据边界：Head 只负责缩小检索范围，选中的变体仍必须由项目代码证据确认；
- 次级指标：禁止“机器分配质量/同机排序质量/竞争压力/并行损失”等无计算口径的
  名称；改为机器选择加工时间增量、最大机器工作负荷、低柔性工序机器负荷等；
- 诊断分离：机器资格一致性、下界可靠性、Oracle 有效性进入独立
  `diagnostic_targets`，不再与待优化指标混排；
- FJSP 复核：旧 CC 结果重建为 6 个可测量次级指标 + 3 个诊断项；关键路径长度
  仅作为 makespan 分解信息，不重复计作次级指标。
- 验证：55 项测试通过；默认长期记忆已重建，classic 回退与 FJSP transport
  Head 单选均通过运行检查。

### 2026-07-26 / module-docs-and-measurable-metrics-v1

- 文档：`docs/` 改为 A–H 模块目录，跨模块总览单独归档；每篇专题文档在开头标注
  主模块、关联模块和文档职责；
- 指标：所有次级指标改为程序固定名称、计算定义和单位，LLM 只做枚举选择；删除
  “后悔值、质量、损失、压力”等不稳定表述；
- 模型：默认 Critic 改为 DeepSeek-v4-pro；Opus 只作人工升级复核，GLM-5.2 因
  本次三次传输失败暂不作为默认；
- 后续状态：该路由已被 `high-recall-impact-review-and-metric-research-v1` 取代，
  本条只保留历史决策；
- 在线测试：DeepSeek Critic 一次通过，18,202 Token、116 秒；Opus Critic
  14,791 日志 Token、208.7 秒、约 $0.469；
- 冗余审计：完整 CC 链失败运行消耗 87,835 Token，其中 20,757 为 Schema repair；
  本轮只报告，没有修改 CC 分批、复读或 checkpoint 结构；
- 验证：56 项测试通过，Markdown 模块标记、链接和指标 JSON 检查通过。

### 2026-07-25 / docs-layout-v1

- 变化：将状态路线图和两层项目图移动到仓库根目录；
- 历史分类：当时建立 `docs/architecture`、`docs/semantics`、`docs/experiments`、
  `docs/guides`；现已由 A–H 模块目录取代；
- 同步：README、维护 Skill、审计脚本和全部 Markdown 相对链接；
- 验证：文档链接检查、架构审计和项目测试；
- 后续：新增文档必须先选择所属一级分类，不再直接堆放到 `docs/` 根目录。

### 2026-07-25 / module-graph-interactive-v1

- 变化：新增根目录可缩放项目图；
- 交互：第一层/第二层切换、滚轮缩放、按钮缩放、拖拽、适应窗口、重置、全屏；
- 约束：交互图与 `PROJECT_MODULE_GRAPH.md` 使用同一两层结构，模块变化时同步更新；
- 验证：浏览器布局与交互检查。

### 2026-07-25 / visual-sync-policy-v1

- 变化：Markdown 不再嵌入静态 PNG 项目图；
- 规则：模块、状态、数据流、实现路径或里程碑变化时，同步更新 Markdown Mermaid、
  交互图和仍作为当前版本使用的其他图片；
- 退役：无法同步的旧截图或导出图必须标记为历史版本或移除；
- 验证：维护 Skill 和架构审计加入图形同步检查。

### 2026-07-25 / staged-semantic-agent-v1

- 变化：新增低成本分片 Navigator、高能力三批 Analyst、证据充分性 MCQ、
  受控 AST 复读工具、压缩短期记忆和最终综合；
- 安全：模型只能申请读取，程序按路径、关系、symbol、重复和预算确定性审批；
- 记忆：SQLite 图谱/FTS 检索可作为批次先验，但禁止作为当前项目 evidence；
- 状态：代码已实现但未接入默认 runtime；尚未执行正式在线质量对照；
- 验证：离线 Fake Provider 端到端、复读审批和秘密/论文排除测试。

### 2026-07-25 / navigator-role-filter-v1

- 触发：FJSP-DRL 预检发现 16 个训练日志会被误送入导航，且环境、训练和评估脚本
  角色过于粗糙；
- 变化：新增 environment/model/training/evaluation 角色，排除训练日志、权重、
  结果表和求解结果 Artifact；
- 安全：评估脚本不再默认视为硬约束测试，训练入口不再默认视为生产入口；
- 验证：新增角色和 Artifact 排除单测；正式 FJSP 在线运行结果待记录。

### 2026-07-25 / opus-routing-and-call-trace-v1

- 决策：强语义分析固定使用 Opus-5；Kimi 仅保留对照，不再作为严格 JSON 主分析；
- 接入：Opus-5 专用通道改由官方 Claude Code CLI 调用，关闭工具和会话持久化，
  并设置逐调用美元预算上限；
- 可观测性：所有 LLM 调用打印并记录 provider、model、task、batch、round、
  repair、耗时、Token、成本和错误类型；
- 安全：事件日志不保存 Prompt、响应正文或密钥，只保存响应 SHA-256；
- 验证：Opus-5 最小探针成功；新增 Provider 配置与 trace 单测，全套 48 项通过；
- 下一步：用固定 FJSP 案例完成一次全链在线验收，并增加批次级 checkpoint/resume，
  避免晚期失败后重付前序调用成本。

### 2026-07-25 / claude-readonly-semantic-skill-v1

- 盘点：本机无全局 Claude Skill；唯一现有项目 Skill 是 Sortie 的
  `sortie3d-harness`；
- 迁移：只提取导航先行、渐进读取、证据账本、确定性门、分层记忆、独立复核、
  有限循环和归档机制，不复制 3D/Blender/Web 领域内容；
- 新增：项目 Skill `.claude/skills/scheduling-code-semantics`；
- 权限：Claude Code 只开放并预批准 `Read/Glob/Grep`，禁止 Bash/Edit/Write；
- 审计：stream-json 记录实际工具名和次数，不记录参数、代码正文或中间响应；
- 验证：真实探针识别 Skill 并只调用一次 Read，49 项离线测试通过；
- 下一步：在固定 FJSP 案例上比较“静态证据包 Opus”和“Skill 只读追踪 Opus”的
  语义准确率、Token、费用和失败率。

### 2026-07-25 / external-code-skill-patterns-v1

- 调研：Anthropic `code-explorer`，Trail of Bits `audit-context-building`、
  `fp-check`、`variant-analysis` 和 Trailmark；
- 吸收：完整执行流、最小必要文件、核心函数 micro-analysis、correction ledger、
  六道语义结论门、exact-to-general 约束变体搜索；
- 控制：不采用全仓库逐行分析，不开放 Web/Bash，不整包复制第三方 Skill；
- 工具：Trailmark/CodeQL/Semgrep 保留为未来 `CodeGraphTool` 可选适配器，默认仍用
  内置 AST + Grep；
- 文档：新增 `docs/modules/module_b_semantics_memory/EXTERNAL_SKILL_PATTERN_REVIEW.md`；
- 下一步：在同一 FJSP 案例上做旧 Skill 与增强 Skill 的配对对照。

### 2026-07-26 / opus-critic-debug-and-conditioned-prior-v1

- 根因：Opus 模型、Token 和 Claude Code 路由可用；实际缺陷是 Provider 未透传
  `--effort`、后分析知识仍受否定句/兄弟变体污染，以及模型返回固定置信度或空扩展
  字段时与严格 Schema 不一致；
- 修复：显式透传推理档；用结构化 family、constraint、environment、objective 做
  后分析知识条件路由；次级目标增加固定三档置信度；只规范化并审计空未知字段；
- 同条件结果：MIMO final 为 3 个次级目标/1 个诊断、11,623 Token、78.5 秒；
  Opus final 为 5 个次级目标/4 个诊断、17,106 Token、66.1 秒；
- 当前判断：Opus final 是当前最完整的高召回专家先验，但尚无人工真值、计算器、
  候选扰动和 Oracle 配对实验，不能称为已验证最优指标集合；
- 产物：`outputs/fjsp_test/FJSP_METRIC_RECALL_COMPARISON_2026-07-26.md` 当时汇总
  14 次运行，现已继续追加高召回实验至 17 次并保持可比性边界。

### 2026-07-26 / high-recall-candidate-pool-v1

- 召回策略：Critic 输入上限由 12 调到 20；当前 FJSP 的 15 个合格候选全部进入
  Prompt，要求保留所有存在合理直接或间接关联的候选，低置信候选不在入口删除；
- 软权重：保留 `confidence=high/medium/low` 作为语义相关性置信层，不把它冒充
  真实因果效应大小；
- 容错：未知扩展字段从运行时控制 Schema 中隔离，同时将路径和值写入
  `schema_extensions`；已知字段类型、候选 ID、引用和重复性仍严格校验；
- 安全：`hard=true → validation=required` 改由程序强制并记录 `safety_overrides`，
  不再由 LLM 决定，也不因可确定的安全字段错误浪费整次输出；
- 在线结果：MIMO 从 15 项保留 8 项；Opus 保留全部 15 项（3 high、6 medium、
  6 low）和 5 个诊断项；两次均形成正式 Artifact；
- 边界：15 项是下一阶段待验证候选池，不是 15 个已确认因素。

### 2026-07-26 / open-world-code-grounded-proposals-v1

- 变化：选择题不再构成封闭世界；Critic 可输出 `novel_secondary_targets` 与
  `novel_diagnostic_targets`，分别承接题库外优化机制和代码特有验证风险；
- 证据门：每项必须绑定现有 constraint ID 和逐字 `file::symbol`；证据不存在、精确
  重名或同轮重复只拒绝该提案并记录原因，不废弃整份报告；
- 候选变化门：新次级目标必须属于 candidate schedule 或 construction trajectory，
  并说明同实例候选变化依据与 intervention handle；实例/实现一致性进入诊断通道；
- 状态门：全部固定为 `proposed`，不得直接写入 active 题库、Factor、CIP 或优化器；
- 在线结果：最终 MIMO 提出 1 个 non-delay 掩码活跃比率和 1 个资格一致性诊断；
  最终 Opus 提出 2 个新次级目标、3 个新诊断，题库内保留 14 项；
- 复核重点：Opus 的“下界 shaping 增量总量”可能因望远镜求和与终态下界重复，虽
  被保留为 low-confidence proposed，但必须先做公式等价审核，不得直接晋级。
