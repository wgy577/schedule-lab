# 长期记忆、分层检索与因果机制目标

> **所属模块**：B — 语义理解与知识记忆  
> **关联模块**：D — 诊断与因果机制  
> **交叉分类**：B × D — 分层记忆、机制先验与因果诊断  
> **文档职责**：定义分层记忆、检索边界及向机制诊断提供先验的方式。

> 文档性质：架构决策、实现契约与持续维护入口  
> 状态：M0–M4 代码已实现但未接入默认优化闭环；M0–M2 已被分阶段语义 CLI 可选读取；M5 尚未实现  
> 最近更新：2026-07-25

## 1. 本轮确定的原则

本项目不把不断增长的调度知识继续塞进单个 JSON 或一次 Prompt。长期记忆采用两种
互补结构：

1. **图谱记忆**保存经过审计的实体、关系、适用范围和冲突，是系统的事实主干；
2. **分层证据记忆**保存论文、代码、测试、案例和实验片段，负责高召回检索。

两者职责不同。图谱回答“哪些关系当前可以被系统采用”，证据记忆回答“原始依据
在哪里、还可能漏掉什么”。可选向量检索只能在已选子树内重排证据，不能直接写入
事实图谱，也不能替代代码、Oracle 或人工审核。

正式优化目标仍是项目声明的目标，例如最小化 makespan \(C_{\max}\)。所谓“次级
目标”统一改称**因果机制目标**：它不是另一个未经声明的正式目标，而是优化动作
影响最终目标的中间通道。

```text
算子/动作
  → 可控因子 X
  → 机制目标 M
  → 调度传播闭包 Ω
  → 最终目标 ΔCmax
```

任何代理指标即使变好，只要 Full Oracle 下的最终目标没有改善，就不能接受候选。

## 2. 双记忆架构

### 2.1 图谱记忆：可信关系主干

第一版使用 **SQLite 属性图表示**，不立即引入 Neo4j：

- 单文件、事务化、可迁移、易备份；
- 适合当前单机研究和可复现实验；
- 节点、边和证据均能保留版本与来源；
- 未来规模或并发确有需要时，再通过稳定接口迁移到图数据库。

当前节点类型：

| 节点 | 含义 |
|---|---|
| `ProblemFamily` | JSP、FSP、FJSP、HFSP 等经典问题族 |
| `Variant` | blocking、no-wait、dynamic、setup 等变体 |
| `EngineeringPattern` | 运输、人员、维护、缓冲、能耗等跨族工程模式 |
| `Constraint` | 可验证约束 |
| `Decision` | 排序、机器分配、开始位置、路径等决策 |
| `StateParameter` | 实例或运行时状态参数 |
| `MechanismTarget` | 空闲间隙、等待、负载不均等机制目标 |
| `Objective` | makespan、tardiness、energy 等正式目标 |
| `Operator` | swap、insert、reassign、destroy-repair 等动作 |
| `Oracle` | 静态、轻量和完整验证器 |
| `Evidence` | 论文、代码、配置、测试和人工结论 |
| `Case` | 项目/实例/incumbent 的作用域 |
| `Experiment` | 干预、结果、成本和后验 |

当前关系类型：

```text
is_a / variant_of / adds_constraint / adds_decision
affects / mediates / controllable_by / validated_by
evidenced_by / contradicts / applies_when / supersedes
```

每个节点和边至少带：

```text
canonical_id, scope, version, source_ids, review_status,
confidence, content_hash, created_at, updated_at,
valid_from, valid_to
```

`confidence` 不能掩盖 `review_status`。模型高置信输出在没有证据时仍是
`proposed`，不能成为 `verified`。

### 2.2 分层证据记忆：高召回检索

证据按语义和粒度分层：

| 层 | 内容 | 首要用途 |
|---|---|---|
| L0 | 问题族和路由摘要 | 快速确定检索子树 |
| L1 | 变体与工程约束模式 | 找项目相对经典模型的增量 |
| L2 | 约束、决策、机制目标、算子和 Oracle | 形成候选解释链 |
| L3 | 论文段落、代码片段、配置、测试、案例 | 证据核验 |
| L4 | 干预结果、失败案例、效应后验和成本 | 经验更新与动作选择 |

当前采用 SQLite 表加 FTS5：

```text
nodes
edges
evidence_chunks
aliases
experiments
memory_versions
```

检索顺序固定为：

1. 精确别名和问题族门控；
2. 图谱邻域扩展；
3. 在选定层级和作用域内执行 FTS5/BM25；
4. 可选 embedding 只对小候选集重排；
5. 组装带来源、冲突和未知项的 evidence pack；
6. LLM 只能基于 evidence pack 做枚举选择与解释。

这可避免向量相似度把不同问题族、旧实验或相互矛盾的事实混在一起。

### 2.3 分阶段语义 Agent 的可选读取

`compile-semantics-staged` 现在可通过 `--memory-database` 在每个语义批次前执行一次
图谱门控 + L0–L4 FTS5 检索。检索结果进入 `LongTermMemoryContext`，其中保留
node ID、chunk ID、层级和冲突；它只帮助模型选择检查方向，不能作为当前项目
`evidence`。最终综合仍必须引用当前仓库文件或受控复读得到的 `ToolEvidence`。

这是“代码已实现但未接入默认 runtime”的可选路径，不证明长期记忆已提高识别质量。
需要在人工确认标签上比较无记忆、JSON 先验和图谱分层记忆后，才能决定是否默认启用。

### 2.4 记忆生命周期

```text
proposed
  → canonicalized
  → source_verified
  → conflict_checked
  → human_or_experiment_approved
  → active
  → superseded
```

- 原始证据不可覆盖，只能追加新版本；
- 结论被推翻时使用 `supersedes`，不物理删除历史；
- 项目事实、通用先验和实验经验必须分库存储、分作用域检索；
- 动态结论绑定 `instance_hash + incumbent_hash + state_hash`，并设置有效期；
- 同一关系的支持证据和反对证据都保留，检索时显式报告冲突。

## 3. 从最终目标发现机制目标

### 3.1 机制目标库

对于 makespan，首批候选机制包括但不限于：

| 机制目标 | 可计算信号 | 典型可控因子 |
|---|---|---|
| 关键资源内部空闲时间 | 相邻工序正空闲间隔之和（time） | 相邻工序次序、上游到达 |
| 实现调度关键路径长度 | 实现析取图最长路径（time） | 只作 makespan 分解和定位 |
| 工序就绪后等待时间 | 就绪到实际开工的时间之和（time） | 上游排序、资源竞争 |
| 资源利用率变异系数 | 同类资源利用率标准差/均值 | 机器分配、批次分配 |
| 序列相关准备时间总和 | setup/changeover 时间之和（time） | 相邻产品/作业次序 |
| 运输引起的等待时间 | 运输与同步等待之和（time） | 车辆、路径、释放顺序 |
| 总阻塞时间 | 占用上游等待下游的时间之和（time） | 邻接顺序、缓冲占用 |
| 资源容量空闲率 | 空闲容量时间/总容量时间（fraction） | 资源分配、先后关系 |

这不是固定清单。项目语义 Agent 可以从证据中提出新增机制，但必须归一化、核验并
补充计算器后才能进入在线选择。

### 3.2 两阶段发现

这里的“两阶段”不是“先选指标、再直接宣布因素”。完整边界为：

**阶段 A：次级目标语义匹配。**

1. 程序根据项目 family、variant、mechanism、decision、role 与 evidence 从 proposed
   题库做多视图召回并按 ID 去重；
2. LLM 只在小批候选中做选择题，输出相关、相邻视图、证据不足或未解决新候选；
3. 程序将语义相关候选与当前可计算候选分流，再执行候选变化门；
4. 输出只叫 `Project-level Secondary Targets`，不能叫 Factor 或 causal driver。

检索分成两个时点：分析前只允许词法检索作为导航提示；语义分析完成后，正式 Critic
先验必须从结构化 `problem_families`、约束种类、环境和目标条件化生成。否定自由文本
不能激活变体，未命中的兄弟变体也不能整包注入。该规则用于限制先验污染和 Prompt
体积，不把知识库内容冒充当前项目证据。

阶段 A 默认“宽进严出”：程序最多提供 20 个结构化候选，LLM 应保留所有存在合理
关联的候选，并用 `high/medium/low` 表达认识置信度。该置信层是后续排序的一项软信号，
不是效应大小；低置信候选只有在阶段 B 的变化门、可控性门、受控干预或人工审查中
失败后才退出，而不是由阶段 A 提前删除。

阶段 A 同时支持开放世界提案，但不直接写入长期 active 记忆。题库外提案必须经过
代码证据门；新次级目标还要通过同实例候选变化声明和干预把手门。通过后状态仍为
`proposed`，等待语义去重、公式等价审核、计算器、真实候选变化、受控干预、Oracle
与人工批准。题库外实现一致性或验证风险单独进入 novel diagnostic，不进入优化排名。

**阶段 B：真正影响因素与交互发现。**

1. 围绕已选 SecondaryTarget，从代码/IR 发现 parameter、state、structure、policy、
   external event 和 interaction；
2. 每个候选绑定具体 symbol、IR 实体、作用域和最小干预；
3. 只有 direct 或经可重放实验验证的 indirect 因素进入在线杠杆；
4. 通过 `operator → factor → mechanism → objective` 路径生成完整候选。

程序随后负责：

1. 计算机制指标；
2. 对局部候选执行受控干预；
3. 通过调度传播或局部重排生成完整可验证候选；
4. 测量 \(\Delta M\)、\(\Delta C_{\max}\)、失败率和验证成本；
5. 更新该作用域下的效应后验；
6. 只把最终目标严格改善且约束通过的候选交给接受器。

不能把

\[
\Delta C_{\max}=\sum_k \text{Effect}(M_k)
\]

当作精确恒等式。机制之间可能交互、重叠或共享路径，实际记录应保留总效应、
可能的中介效应、交互项和残差。

## 4. 因子进入优化的资格门

### 4.1 候选区分度是硬门

用户提出的判断是正确的：一个量要用于区分同一实例的候选，首先必须在这些候选
之间发生变化。

```text
同一 instance + 同一环境状态下：
  candidate-constant
    → 不进入候选优化权重
    → 可保留为上下文、可行性条件或 Oracle 输入

  decision-dependent / mixed / exogenous-changing
    → 才继续检查可控性和目标通路
```

“影响绝对工期”不等于“能帮助当前优化器挑出更好的候选”。

### 4.2 可控性分类：暂定接口

第一版建议保留五类：

| 类别 | 暂定定义 | 是否可成为优化杠杆 |
|---|---|---|
| `direct` | 当前动作显式修改该变量 | 是 |
| `indirect` | 动作修改其上游，重放后该变量可重复地变化 | 是，但需干预证据 |
| `fixed` | 同实例候选间不变 | 否 |
| `external` | 只能由环境事件或外部 Oracle 改变 | 通常否 |
| `unknown` | 证据不足 | 不自动使用 |

这里不把“图上可达”直接视为 `indirect`。建议最终规则为：只有至少一次可重放的
合法干预证明 `action → factor`，才升级为间接可控；否则保留为
`hypothesized_indirect`。这项规则在实现前仍可讨论和调整。

### 4.3 完整资格链

一个因子进入在线优化优先级，依次通过：

1. **变化门**：能否区分当前候选；
2. **可控门**：是否 direct 或有证据的 indirect；
3. **机制门**：能否连接某个可计算机制目标；
4. **传播门**：能否生成完整、合法的候选；
5. **效应门**：受控干预是否改善最终目标；
6. **成本门**：预期收益是否值得验证时间与 Token；
7. **安全门**：所有硬约束仍必须验证。

## 5. 权重如何产生

LLM 的档位只表示初始关注顺序，不是真实效应。建议把在线优先级拆成可审计组件：

```text
Priority =
  candidate_variation_gate
  × controllability_gate
  × expected_objective_gain
  × validity_probability
  × evidence_reliability
  × budget_value
```

其中：

- `candidate_variation_gate` 和不满足可控性的门可直接为 0；
- `expected_objective_gain` 来自同作用域的受控局部实验后验；
- `validity_probability` 来自静态/轻量/完整 Oracle 历史；
- `evidence_reliability` 由代码、测试、论文、人工和实验的证据等级决定；
- `budget_value` 综合求解时间、Oracle 时间、Token 和失败成本。

数值需要按实例规模和指标尺度归一化。没有实验数据时保持 `null/pending`，不能用
LLM 的语言置信度伪装成数值效应。

### 5.1 剩余问题的处理方案

| 问题 | 处理方式 |
|---|---|
| binding/slack | 由确定性排程分析器计算，不问 LLM |
| 因子交互 | 只对高优先因子做 top-k 成对/条件干预，避免组合爆炸 |
| 传播范围 | 图可达性提出闭包，实际 replay 记录闭包外变化进行校准 |
| 关键阶段/位置 | 关键路径、瓶颈、sink stage 计算器给出 |
| 证据强度 | LLM 选择证据类型，程序检查引用并映射审计等级 |
| 动态权重 | 绑定 problem/instance/incumbent/state hash，不保存为全局常数 |
| Oracle 昂贵 | 多保真筛选，Full Oracle 只验证高采集价值候选 |
| 不确定项 | 保留 unknown 和 abstain，不强制中间分 |
| 代理指标作弊 | 最终接受始终回到正式目标和 Full Oracle |
| 校准漂移 | 定期比较先验档位、后验效应和真实改善，失配则降权 |

## 6. LLM 选择题协议

后续接入时让高能力模型依次回答：

1. 当前项目属于哪个问题族或组合；
2. 命中哪些经典变体和工程模式；
3. 当前损失最可能由哪些机制目标产生；
4. 因子在同实例候选间是
   `constant / decision-dependent / exogenous / mixed / unknown`；
5. 可控性是
   `direct / hypothesized-indirect / verified-indirect / fixed / external / unknown`；
6. 因果路径是
   `direct / mediated / speculative / none / unknown`；
7. 证据是
   `solver-code / validator-code / test / data / paper / prose / none`；
8. 应从批准库选择哪个干预、指标计算器和验证器；
9. 哪些问题必须交给人工确认。

自由文本只用于简短理由和待确认问题。候选集合、数值效应、硬约束结论和最终接受
不由 LLM 单独决定。

## 7. 实施顺序

| 阶段 | 内容 | 状态 |
|---|---|---|
| M0 | 冻结节点、边、证据、版本和作用域 Schema | 代码已实现但未接入 |
| M1 | 将现有 `scheduling_families.json` 导入 SQLite 图谱 | 代码已实现但未接入 |
| M2 | 建立 L0–L4 分层 FTS5 检索和 evidence pack | 代码已实现但未接入 |
| M3 | 建立 makespan 机制目标库与确定性计算器 | 代码已实现但未接入 |
| M4 | 建立受控干预、效应记录和后验更新 | 代码已实现但未接入 |
| M5 | 完成四问题族离线校准后接入 CIP/Agent | 尚未实现 |

M0–M4 已具备独立 CLI、持久化 Artifact 和单元测试，但默认语义编译、CIP 和 Agent
尚未消费这些结果。未经校准的权重不得直接过滤硬约束、删除候选或控制最终接受。

当前实现位置：

- `src/causal_schedule_lab/storage/graph_store.py`；
- `src/causal_schedule_lab/storage/memory.py`；
- `src/causal_schedule_lab/mechanisms.py`；
- `src/causal_schedule_lab/knowledge/mechanism_targets.json`；
- `tests/test_memory_and_mechanisms.py`。

## 8. 尚未冻结的决策

1. `verified-indirect` 需要一次成功 replay，还是需要跨实例重复证据；
2. 图谱迁移到专用图数据库的规模阈值；
3. 机制后验采用分层贝叶斯、稳健频率统计还是二者并存；
4. 高成本 Oracle 的采集函数如何同时计入时间、Token 和失败风险；
5. 交互效应只测成对因子，还是允许 Agent 对少量三元组合提出实验；
6. 不同问题族之间哪些经验允许迁移，哪些必须隔离。

这些项应通过接口、消融和真实案例决定，不在数据出现前凭直觉写死。
