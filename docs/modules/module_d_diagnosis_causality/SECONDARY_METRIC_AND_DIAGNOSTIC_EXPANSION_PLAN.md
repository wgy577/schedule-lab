# 次级指标与诊断项扩充研究任务书

> **所属模块**：D — 诊断与因果机制  
> **关联模块**：B — 语义理解与知识记忆；H — 实验记忆与统计  
> **交叉分类**：B × D × H — 文献语义、机制诊断与证据验收  
> **文档职责**：规定资料范围、LLM 提取协议、候选分类、验收门和知识库更新方式。  
> **维护触发**：新增调度问题族/变体、次级指标、诊断项、计算器或实验证据时更新。  
> **建立日期**：2026-07-26  
> **当前状态**：Round 3–6 proposed 知识已统一加载；确定性多视图召回与 Token 有界
> 选择包已接入可选影响 Critic。候选仍未自动进入优化器，诊断大库尚未接入动态选择。

## 1. 为什么现在必须扩充

当前 active 正式目录仍包含 11 个次级指标和 4 个诊断项。它足以支撑经典 FJSP 的第一轮机制
分析，但不足以覆盖 JSP、FSP、FJSP、HFSP 及其 blocking、no-wait、setup、运输、
辅助资源、动态和随机变体。

这里属于整个优化链的高召回入口：漏掉关键中间机制，后续 Factor、CIP、局部算子和
Oracle 即使实现正确，也只会在错误或残缺的搜索空间内工作。因此采取以下原则：

1. `proposed` 阶段追求高召回，允许保留有证据但尚未验证的候选；
2. `active` 正式目录追求高精度，必须有固定公式、适用边界和可执行计算口径；
3. DeepSeek 暂停承担候选删除、降权和最终审核，影响审核 CLI 直接拒绝该路由；默认审核使用 Opus；
4. 低成本模型只可做文献导航、目录整理和重复项预聚类，不能否决候选；
5. 论文中的“优化目标”不自动等于本项目的“次级机制指标”。

## 2. 需要阅读哪些资料

### 2.1 第一层：问题族总览综述

目的不是抄一张目标列表，而是建立每个问题族的标准假设、主要变体、常见决策和
指标所在层级。

每个问题族至少收集 2–3 篇互相独立的综述：

| 资料组 | 要覆盖的内容 | 起始资料 |
|---|---|---|
| FJSP | criteria、constraints、configurations、dynamic/uncertain extensions | [Dauzère-Pérès et al., FJSP review](https://doi.org/10.1016/j.ejor.2023.05.017) |
| HFSP | stage、parallel machines、setup、buffer、machine eligibility | [Ruiz & Vázquez-Rodríguez, HFS review](https://doi.org/10.1016/j.ejor.2009.09.024) |
| Blocking FSP | blocking location、buffer capacity、flow time、cycle time | [Miyata & Nagano, blocking FSP review](https://www.sciencedirect.com/science/article/abs/pii/S0957417419304774) |
| No-wait/setup FSP | forced delay、no-wait feasibility、setup type、sequence effects | [Nagano et al., no-wait/setup review](https://www.sciencedirect.com/science/article/pii/S2468601824000403) |

JSP 与 FSP 经典定义可由标准教材/综述作为基线，但正式机制仍须回到论文中的公式或
可执行定义，不能只引用二手解释。

### 2.2 第二层：工程变体专题综述

每类至少选择 1 篇综述和 2 篇有完整数学模型的代表性论文：

- 运输车辆/AGV/机器人：loaded travel、empty travel、pickup waiting、vehicle
  conflict、machine–vehicle synchronization；起始资料为
  [FJSP with transportation vehicles review](https://doi.org/10.1631/FITEE.2300795)。
- blocking、有限 buffer、no-wait、no-idle；
- sequence-dependent setup、family setup、batching；
- 人员、工具、模具、维护、换班等辅助资源同步；
- release、due date、动态到达、机器故障和重调度；
- 随机加工时间和鲁棒调度；
- 能耗/峰值功率只在它与资源开关、速度或批次决策共同影响 makespan 时作为机制
  候选，否则保留为独立目标，不强行塞入 makespan 次级指标。

动态与随机场景的起始资料：

- [Vieira et al., efficiency and stability in dynamic rescheduling](https://doi.org/10.1016/j.cie.2003.09.007)；
- [Surrogate Measures for Robust Stochastic JSSP](https://www.mdpi.com/1996-1073/10/4/543)。

### 2.3 第三层：机制与邻域的原始论文

综述负责发现术语，原始论文负责确认“为什么该量能定位 makespan 改进位置”：

- bottleneck identification 与局部重优化：
  [Adams, Balas & Zawack, Shifting Bottleneck](https://doi.org/10.1287/mnsc.34.3.391)；
- workload、utilization、throughput 等多属性瓶颈指标：
  [Schedule-based execution bottleneck identification](https://www.sciencedirect.com/science/article/pii/S0360835216301929)；
- critical path / critical block 邻域、机器序列交换、插入与重排；
- lower bound 分解：job-chain、machine-load、stage-load、setup、transport 和 resource
  synchronization 下界；
- 对每种局部算子，记录它直接改变的 Decision、首先改变的机制指标和传播闭包。

### 2.4 第四层：真实工程案例

优先选择同时公开以下至少三项的论文或项目：数学模型、实例/代码、结果表、消融或
仿真。工程案例用于发现综述没有标准化命名的交叉机制，例如：

- `machine assignment × AGV routing`；
- `setup family × batch release`；
- `maintenance window × critical machine sequence`；
- `worker eligibility × machine eligibility`；
- `buffer capacity × blocking propagation`。

只有摘要、没有公式或没有上下文的文章只能进入检索线索，不能直接进入正式目录。

## 3. LLM 每篇资料必须总结什么

每个候选必须生成一条结构化记录，不允许只写自然语言摘要：

| 字段 | 说明 |
|---|---|
| `source_id` | DOI/arXiv/稳定 URL 与页码、公式号或表号 |
| `problem_family` | JSP/FSP/FJSP/HFSP，只能多选明确出现的族 |
| `variant_heads` | blocking、no_wait、transport 等显著 Head；无证据选 classic |
| `candidate_role` | secondary_metric / diagnostic / factor / decision / constraint / primary_objective / reject |
| `canonical_name` | 正常调度术语，不使用“质量、效率、压力、损失”等空泛名称 |
| `source_term` | 论文原术语，便于别名与翻译审计 |
| `formula` | 原公式或可无歧义复现的伪公式 |
| `variables` | 每个变量的实体、时间点和数据来源 |
| `unit` | time、count、fraction、energy、weighted_time 等 |
| `aggregation_scope` | operation/job/machine/stage/vehicle/system/pairwise experiment |
| `desired_direction` | decrease/increase/stabilize/context_dependent |
| `candidate_variation` | constant/decision_dependent/exogenous/mixed/unknown |
| `controllability` | direct/indirect/fixed/external/unknown |
| `objective_path` | Decision → Factor/Interaction → Metric → makespan 的明确链条 |
| `applicability` | 必须具备哪些约束、资源和数据才可使用 |
| `counterexample_boundary` | 什么条件下该指标与 makespan 无关、反向或退化为常量 |
| `calculator_inputs` | 前端/IR/Oracle 需要提供的字段 |
| `evidence_grade` | survey_map/formula_verified/code_backed/intervention_supported |
| `confidence` | high/medium/low，不能替代 evidence grade |

## 4. 尽量让 LLM 做选择题

### 4.1 第一问：角色分类

只能选择一个：

```text
secondary_metric / diagnostic / factor / decision /
constraint / primary_objective / reject / unresolved
```

判定规则：

- 能对一个候选排程或成对候选计算标量，并位于 makespan 上游：
  `secondary_metric`；
- 检查数据、模型、下界、Oracle 或实验是否可信：`diagnostic`；
- 被决策改变并进一步影响指标的参数、状态或结构：`factor`；
- 分配、排序、插入、路由、批次、时间选择：`decision`；
- 只规定可行域：`constraint`；
- 只是论文最终优化目标且没有中介含义：`primary_objective`；
- 与调度候选无关或只是训练硬件/对比算法：`reject`。

### 4.2 第二问：指标家族

候选次级指标从以下大类选择；具体分支必须逐项命名，不能输出“各类变体”：

1. 时间流与等待分解；
2. critical path / critical block / slack 结构；
3. 瓶颈资源、工作负荷与阶段同步；
4. 机器资格、柔性与路由选择；
5. setup/changeover/batch；
6. blocking/buffer/no-wait/no-idle；
7. transport/AGV/mobile resource；
8. 人员、工具、维护等辅助资源同步；
9. 动态重调度稳定性；
10. 随机环境鲁棒性；
11. decoder/solver 造成的可行插入差异；
12. other / unresolved。

### 4.3 第三问：是否值得进入候选库

逐项回答 yes/no/unknown：

- 是否有精确公式或可执行定义？
- 是否在同一实例的候选之间可能变化？
- 是否不是主目标的同义重复？
- 是否能追踪到至少一个可修改 Decision？
- 是否说明适用变体和失效边界？
- 是否具有程序可获得的输入？
- 是否与已有指标不完全等价？

任一关键项为 `unknown` 时保留在 `proposed`，不能删除，也不能进入 `active`。

## 5. 当前优先补充的候选方向

以下只是文献检索槽位，不是已经批准的正式指标：

### 5.1 次级指标候选

- critical block 数量、长度、边界空闲与可移动性；
- operation/job slack、关键操作比例和近关键路径负荷；
- 关键阶段内部空闲、stage starvation、stage clearance gap；
- job-chain waiting、machine-queue waiting、同步等待的可避免/不可避免分解；
- bottleneck workload excess、stage workload imbalance、eligibility concentration；
- setup 次数、family switch 次数、anticipatory/non-anticipatory setup 等待；
- buffer occupancy、blocking propagation length、no-wait forced start delay；
- AGV 空驶、载货行驶、取货等待、车辆冲突等待、运输同步等待；
- worker/tool/maintenance synchronization waiting；
- rescheduling start-time deviation、sequence deviation、recovery duration；
- robustness surrogate、expected degradation、tail-risk/CVaR 类指标。

### 5.2 诊断项候选

- 项目变体识别完整性；
- 指标公式可计算性和数据覆盖率；
- 同实例候选变化性/非退化性；
- Agent/优化器直接或间接可控性；
- 与 makespan 的中介路径完整性；
- 与已有指标的等价、冗余和共线性；
- 方向符号在实例、规模和负载区间内的稳定性；
- lower bound 的 admissibility 和分解完整性；
- Validator 对所有硬约束的覆盖；
- Oracle 的确定性、保真度和版本一致性；
- 指标单位、归一化和跨规模可比性；
- 干预实验的混杂、泄漏和配对有效性；
- 文献公式、代码实现和项目数据之间的一致性。

## 6. 执行工作流

```text
综述导航（低成本模型可做，不得删候选）
  → Opus 高召回提取候选和证据定位
  → 程序做 Schema、枚举、引用、单位和重复项检查
  → 第二个独立 Opus 上下文做公式/边界复核
  → 候选进入 proposed
  → 对真实 schedule 执行 computability + variation gate
  → 做局部干预与 Oracle 配对实验
  → 人工批准 active / 保留 proposed / rejected
  → 再扩充 Python enum、固定目录、计算器和测试
```

当前禁止让单个模型同时“发现候选并决定删除候选”。发现与否决必须分离；否决必须
引用具体公式、代码事实或实验结果。

## 7. LLM 交付物

每一轮资料扫描必须交付：

1. `source_manifest.json`：资料、版本、范围、是否获得全文；
2. `candidate_metrics.jsonl`：逐候选结构化记录；
3. `candidate_diagnostics.jsonl`：逐诊断项记录；
4. `alias_and_equivalence.json`：术语别名、可能重复和明确不等价关系；
5. `coverage_matrix.md`：问题族 × 变体 × 指标家族覆盖；
6. `unresolved_questions.md`：缺公式、缺代码、边界冲突和需要人工判断的内容；
7. `promotion_report.md`：为什么进入 proposed/active/rejected，附证据。

## 8. 验收标准

资料扫描完成不等于目录扩充完成。正式新增一项必须同时满足：

- 有稳定术语、公式、单位和聚合范围；
- 有明确适用的 family/variant Head；
- 能说明它不是 makespan 的同义重复或实例常量；
- 能连接到至少一个 Decision，并声明直接/间接可控；
- 有计算输入和缺失数据行为；
- 有至少一个正例和一个失效/边界例；
- 有单测、同实例候选变化测试和 Oracle 配对实验计划；
- 经独立高能力模型复核和人工审批。

在真实干预证据形成前，只能称为“文献支持的机制候选”，不能称为已发现的真实因果
影响因素。

## 9. Round 4 proposed 知识物化

Round 2/3 的外部扩充结果已经以真实 ID 导入项目：

```text
src/causal_schedule_lab/knowledge/secondary_metrics/
├── round3/   # 55 个 canonical candidates、25 个 diagnostics、623 条关系及来源队列
└── round4/   # 条件建议、可计算性契约、能力缺口组和参考场景
```

代码入口：

`src/causal_schedule_lab/secondary_metric_knowledge.py`

可重复生成命令：

```bash
python3 scripts/build_round4_knowledge.py
```

本轮已机械确认：

- 55 个 canonical candidate 保持唯一且全部为 `proposed`；
- 25 个 diagnostic 使用独立 ID 空间，不冒充 55 个指标实体；
- 623 条 membership 全量物化且 `(candidate_id, view_id)` 唯一；
- 44 个 gap 全部归入 9 个允许重叠的能力组；
- 55 个候选全部具有 `semantic_candidate` 与 `computability_requirements` 分离契约；
- 21 个场景只标记为 `design_reference_not_independent_truth`。

该段是 Round 4 历史状态；当前运行状态见下一节。

## 10. Round 5/6 导入与确定性召回运行链

2026-07-26 已将两轮增量包按原始文件完整导入：

| 版本 | 指标 | 诊断 | 原始关系 | 重点范围 |
|---|---:|---:|---:|---|
| Round 3/4 基线 | 55 | 25 | 623 | JSP/FSP/FJSP/HFSP 通用机制与既有工程变体 |
| Round 5 | +12 | +5 | +171 | 批处理、lot streaming、WIP、人员/夹具、AGV 充电、装配同步 |
| Round 6 | +14 | +6 | +257 | 分布式、多工厂、跨厂运输、动态事件与重调度 |
| Legacy 兼容 | +4 | 0 | +6 | 保留原 active 11 中无法与基线公式等价合并的四项 |

统一目录现在包含 `85` 个 canonical metric、`36` 个 diagnostic 和 `1056` 条去重关系。
Round 5 有一组相同 `(candidate_id, view_id)`、相同角色与方向但理由不同的记录；运行时
合并为一条关系并保留 `alternate_reasons`，原始文件不改写。

原 active 11 项通过 `compatibility/legacy_active_metric_bindings.json` 全量映射：语义与
公式等价者复用已有 canonical ID，不等价者新增独立 ID，禁止用近似指标冒充等价项。

当前召回链：

```text
项目 SemanticAnalysis
  → family 硬过滤
  → 显式 variant activation gate
  → mechanism / modifiable decision / evidence 确定性排序
  → required_ir_fields 缺口标记（不凭空推断）
  → top 20 + 每批上限 12 的紧凑 JSON 高召回候选包
  → Opus Critic 只返回候选包中的 candidate_id
  → 若代码出现目录缺口，分别提出 novel metric / novel diagnostic
  → 程序校验 constraint、file::symbol、候选变化、干预把手、精确去重
  → 仅写入 proposed 审核池
  → 程序白名单、重复 ID、source constraint 与 hard validation 校验
  → pending_human_review SecondaryTargetChoice
```

这条链不随模型采样改变召回结果；同一 `SemanticAnalysis` 会产生同一 fingerprint、顺序
和候选包。LLM 不再阅读 85 项正文，只阅读最多 20 项，因此题库扩容不会线性增加 Prompt。
选择阶段以召回优先，允许低置信候选进入下一层；`confidence` 只作为未校准软权重，
不能替代真实候选变化、干预效应或人工晋级门。

仍未完成：

- 36 项诊断大库目前可读取，但 D10 仍使用固定诊断枚举；
- `required_ir_fields` 目前依据语义证据中显式字段标记，不等价于 adapter 实际字段审计；
- 尚无跨问题族人工标签的 Recall、Precision、NDCG 与校准实验；
- proposed 指标未通过 variation、干预、Oracle 和人工批准，不得自动影响优化器。
- 题库外提案尚缺公式语义等价/化简器；类似望远镜求和与已有终态指标重复的问题必须
  在晋级前显式检查。
