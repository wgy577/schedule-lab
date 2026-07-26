# 第一阶段收口与第二阶段 Agentic 排程优化实施说明

> **所属范围**：A × B × C × D × E × F × G × H  
> **文档职责**：向不了解本项目的 LLM、Agent 或协作者完整说明第一阶段已经产出什么、
> 尚未证明什么，以及第二阶段如何围绕甘特图、真实影响因子、局部算子和项目求解器推进。  
> **当前软件基线**：`0.7.4`  
> **阶段状态**：第一阶段暂时封板；第二阶段尚未执行真实批量因子干预或 Agentic RL 训练。  
> **最近核验**：2026-07-26

## 0. 一页结论

项目的最终任务是：在保持项目全部调度约束的前提下，改进已有甘特图/排程，首先关注
makespan，再按项目正式目标处理其他目标。我们优先复用目标项目已有的求解器、环境、
Validator 和 Oracle；LLM/Agent 负责诊断“哪里值得改、可能由什么因素造成、选什么算子、
释放多大局部区域、是否继续”，不直接凭语言生成最终甘特图。

优先级固定为：

1. **P0：排程与甘特图改善。** 关键资源空闲、等待、机器分配、关键块次序、局部传播
   和最终 makespan 是主要研究对象。
2. **P1：可行性与比较口径守门。** 资格集合、Oracle 状态、独立 Validator 等必须在
   候选验收时检查，但它们不是主要优化目标。
3. **P2：实现/训练代码诊断。** 只在它直接限制可达排程、污染目标或妨碍求解器调用时
   处理；不把一般代码优化放在甘特图优化之前。
4. **P3：论文、展示和非运行信息。** 只作解释、溯源或实验背景，不控制在线搜索。

第一阶段完成的是“项目理解与次级目标入口”，不是最终因素发现。当前手里已有可靠的
项目语义、约束/决策/Oracle 证据、统一候选题库、高召回选择结果和开放世界提案；真正
的 Factor、Interaction、有效算子及效应大小要在第二阶段通过排程候选和 Oracle 实验得到。

## 1. 第一阶段解决了什么

第一阶段回答四个问题：

1. 目标项目到底是什么调度问题；
2. 当前执行路径中有哪些目标、约束、可修改决策和 Oracle；
3. 哪些可测量次级目标可能帮助解释或改进甘特图；
4. 题库之外是否存在由项目代码特有实现产生的新机制或验证风险。

执行链为：

```text
仓库文件清单与角色过滤
  → 低成本 Navigator 建立代码导航
  → Opus 按环境/约束/决策三批分析
  → 证据不足时申请精确 symbol/callee 复读
  → 程序验证 file/symbol 引用并综合 Project Semantics
  → 问题族/变体/工程模式条件检索
  → 85 项指标目录确定性召回至最多 20 项
  → Opus 高召回选择 + high/medium/low 置信分层
  → 题库外 metric / diagnostic 开放提案
  → 代码证据、候选变化、干预把手、去重和 proposed 门
```

第一阶段没有训练新模型，没有运行排程改进实验，也没有证明任何指标是 makespan 的
真实原因。所有 LLM 输出仍是 `structured_llm_prior` 或 `proposed`。

## 2. 当前项目语义基线

当前在线案例是 FJSP-DRL/DANIEL 官方实现。保存的语义 Artifact 为：

`outputs/fjsp_test/fjsp_opus5_staged_semantics.json`

已提取并审计：

| 内容 | 当前结论 |
|---|---|
| 问题族 | 经典静态、确定性 FJSP；连续时间、离线构造式求解 |
| 正式目标 | 唯一主目标为最小化 makespan |
| 训练信号 | `op_ct_lb` 完工下界差分 shaping；不是独立调度目标 |
| 决策 | 选择作业当前工序、选择兼容机器、选择求解策略 |
| 非决策 | 开工时间由作业就绪和机器空闲时间派生 |
| 硬约束 | 作业内工序优先、机器资格、每工序恰选一机、同机不重叠、release=0 |
| 项目特有限制 | non-delay 近似动作掩码收窄；不是经典 FJSP 的硬约束 |
| 主要 Oracle | OR-Tools CP-SAT，默认 1800 秒上限 |
| 环境 Oracle | 项目 step/mask 构造环境，负责生成可行排程 |
| 独立 Validator | 未发现；项目只落盘 makespan 与耗时 |
| 已知风险 | 环境归一化后的资格判定与 OR-Tools 原始 `op_pt!=0` 可能不一致 |

语义结果包含 6 条约束、4 个决策、3 个 Oracle 和 5 个未决问题，原完整链累计
67,492 Token。所有结论均带文件与 symbol 证据。

## 3. 第一阶段手里的实物

### 3.1 通用知识与稳定选择题

- 85 个 `proposed` secondary metric；
- 36 个 diagnostic；
- 1056 条 family/variant/mechanism/decision 等关系；
- 原11项活动指标兼容映射；
- JSP、FSP、FJSP、HFSP 及显著工程变体知识；
- 分析前词法导航、分析后结构化条件路由；
- 最多20项高召回候选包、ID白名单、置信分层和完整审计。

题库源位于：

```text
src/causal_schedule_lab/knowledge/secondary_metrics/
src/causal_schedule_lab/knowledge/scheduling_families.json
src/causal_schedule_lab/knowledge/variant_heads.json
```

### 3.2 当前 FJSP 的14个目录内次级目标

正式开放世界结果：

`outputs/fjsp_test/fjsp_opus5_open_world_qualified_final.json`

| 置信度 | 次级目标 |
|---|---|
| high | 机器选择加工时长机会损失、最大机器工作负荷、工序间等待总量、机器空闲总量 |
| medium | 稀缺资格负荷、机器工作负荷变异系数、机器队列等待总量、关键资源内部空闲时间 |
| low | 瓶颈负荷超额、资源容量空闲率、资源利用率变异系数、分配集中度 HHI、平均连续活跃时长、在制品占用面积 |

这些是“排程诊断入口”。第二阶段不应一次实现全部计算器，而应优先处理能够从甘特图
和统一 IR 稳定计算、与当前可修改决策关系清楚的 high/medium 项。

### 3.3 两个题库外次级目标

1. **non-delay 掩码剪除对数量累积**，medium：沿构造轨迹累计被项目 non-delay
   收窄剪掉的原本资格/前驱可行 `(job,machine)` 对。它直接指向可达排程空间，是当前
   最值得进入第二阶段的开放提案。
2. **下界 shaping 增量总量**，low：可能因正增量求和产生望远镜效应而与终态下界
   重复。必须先做符号化简和轨迹反例测试；未通过前不进入优化控制。

### 3.4 诊断项

固定诊断：资格完整性、目标下界可信度、Oracle参照有效性、Validator覆盖。

开放诊断：

- 环境归一化资格集合与原始 `op_pt` 资格集合一致性；
- CP-SAT 求解状态和1800秒时限触发记录；
- 缺失独立排程可行性校验器。

这些诊断的角色是防止错误候选、错误基准或不可行甘特图进入实验。它们可以阻止一次
候选验收，但不应排在排程因素/算子搜索之前充当主要优化方向。

### 3.5 当前可复用代码能力

| 能力 | 接口 | 当前状态 |
|---|---|---|
| 统一 Problem/Schedule IR | `ir.py`、`io.py` | 主闭环已运行 |
| 调度异构图 | `graph.build_scheduling_graph` | 主闭环已运行 |
| CIP 召回/排序 | `cip.CausalCoreDiscoverer`、`CIPRanker` | 规则主链已运行 |
| 机制计算与资格门 | `mechanisms.measure_mechanisms`、`qualify_factor` | 代码已实现但未接入默认控制器 |
| 算子动作封装 | `operators.ActionIndex`、`build_intervention` | 代码已实现 |
| 确定性邻域组合 | `search.DeterministicNeighborhoodPortfolio`、`TabuMemory` | 代码已实现 |
| 局部精确修复 | `repair.GenericCPSATRepairGenerator` | 主闭环已运行 |
| 条件生成器 | `conditional_generator.SolverBackedConditionalGenerator` | solver fallback 可用，学习模型未训练 |
| 层级 Agent 动作 | `agentic_rl.HierarchicalActionSpace`、`decode_action` | 代码已实现但未训练 |
| PPO/BC | `agent.MaskedPPOAgent`、`advantage_weighted_behavior_cloning_loss` | 代码已实现但无正式 checkpoint |
| 四级验证 | `core_validation.validate_schedule`、`validation.MultiFidelityValidator` | 通用主闭环已运行；项目 Full Oracle 外接 |
| 接受/回退 | `controller.AgenticImprovementController` | 主闭环已运行 |
| 后验与统计 | `posterior.MultiFidelityPosterior`、`statistics.py` | 代码已实现，尚无真实大规模数据 |

## 4. 第一阶段明确没有完成的内容

- 没有找到最终 Factor 或 Interaction；
- 没有证明14+2个次级目标中哪个最能促进 makespan 改善；
- 没有为当前项目实现这些指标的全部 adapter/calculator；
- 没有在同一实例多个候选甘特图上验证候选变化；
- 没有批量执行受控局部干预；
- 没有训练 Agentic RL、CIP GNN 或条件生成模型；
- 没有证明跨 JSP/FSP/FJSP/HFSP 泛化；
- 没有把开放提案自动晋级到 active 知识或优化器；
- 没有建立题库外公式的自动等价/化简器。

## 5. 第二阶段的正式目标

第二阶段要回答：

> 对当前 incumbent 甘特图，哪个可修改因素导致了关键资源空闲、等待或负荷问题；
> 应使用哪个局部算子、释放多大传播闭包、调用哪种求解器预算，才能得到完整可行且
> makespan 严格改善的排程？

推荐闭环：

```text
实例 + incumbent + 甘特图
  → 计算高/中置信次级指标
  → 定位关键资源、关键块、等待传播与异常区段
  → Agent 提出 Factor / Interaction 假设
  → Agent 选择 operator × closure × solver budget
  → 项目求解器或 CP-SAT 生成完整候选排程
  → Code/Static/Light/Full Oracle 验证
  → 测量 Δmetric、Δmakespan、可行性、时间和失败类型
  → 更新作用域后验和禁忌记忆
  → 严格改善则接受，否则回退
```

## 6. 最终影响因子如何定义

Factor 不能只是“机器分配质量”或“等待很大”。每个 Factor 必须绑定可执行实体：

| 因子类型 | 示例 | 必须绑定 |
|---|---|---|
| decision | 某工序机器选择、同机相邻次序 | operation/resource IDs 与 IR 字段 |
| state | 某关键时刻机器空闲、工序未就绪 | schedule timestamp 与节点集合 |
| structure | 关键块、瓶颈阶段、汇聚资源 | 图节点/边与关键路径位置 |
| policy | non-delay mask、dispatching rule | 具体策略参数或动作过滤规则 |
| parameter | 局部释放半径、阈值、求解预算 | 可重放配置键 |
| interaction | 机器分配 × 上游排序、mask × 关键块 | 两个以上实体及联合干预 |

进入优化候选前至少满足：同实例候选可变、直接或验证间接可控、能生成合法完整排程、
能测量次级目标与 makespan 的联合变化。

## 7. Agent 与工具的职责划分

不建议让一个自由 Agent 同时阅读代码、改排程和宣布结果。推荐一个编排 Agent 加五类
受控角色/工具：

1. **Schedule Diagnosis Agent**：读取统一 IR、甘特图和次级指标，定位关键区段；
2. **Factor Scout Agent**：提出绑定实体的 Factor/Interaction 假设和最小干预；
3. **Operator Planner Agent**：从合法算子库选择算子、闭包等级和预算；
4. **Solver Tool**：调用项目原求解器、局部 CP-SAT、VNS/ALNS/Tabu 或条件生成器；
5. **Oracle/Critic Tool**：验证硬约束和正式目标，不接受语言模型自报结果；
6. **Experiment Memory**：保存全部成功/失败干预、成本和作用域后验。

LLM/Agent负责提案和选择，确定性程序负责执行，Oracle负责裁决。

## 8. 第二阶段 Agentic RL 设计

### 8.1 状态

状态至少包含：

- 实例规模、问题族和变体；
- incumbent 正式目标与甘特图；
- 关键路径/关键资源/关键块/空闲/等待/负荷特征；
- 当前 SecondaryTarget 值和置信层；
- CIP `(D,R,P,Ω)`；
- 已尝试的 factor/operator/closure、收益和失败；
- tabu memory、效应后验、剩余时间/Oracle/Token预算。

### 8.2 动作

完整动作建议扩展为：

```text
(secondary target,
 factor hypothesis,
 operator,
 closure level,
 solver/control action,
 budget tier)
```

其中 solver/control action 包括 retry、expand、next CIP、backtrack、调用 Full Oracle、
停止局部搜索和停止全局搜索。

### 8.3 算子组合

优先实现可解释、局部和可冻结的算子：

- 关键机器块相邻交换/插入；
- 关键工序重排；
- 可替代机器重分配；
- 关键路径前驱释放和局部后缀重排；
- bottleneck/等待导向 destroy-repair；
- bounded VNS、ALNS 与 tabu；
- 闭包外冻结的 CP-SAT 精确修复；
- FJSP 项目特有的 non-delay mask 放宽/阈值消融工具。

具体问题族只开放合法算子。例如 JSP 不开放机器重分配，FJSP 开放；FSP/HFSP 的
阶段/并行机算子按 Adapter capability 声明启用。

### 8.4 奖励

奖励不能让模型用次级目标替代正式目标。建议使用词典序接受门，并让训练奖励包含：

```text
正式目标严格改善奖励
+ 可行候选奖励
+ 次级目标朝预期方向变化的 shaping
- Oracle/求解时间成本
- 无效候选、闭包外变化和重复动作惩罚
```

只有通过 Full 验证且正式目标严格改善的候选才能更新 incumbent。诊断项不进入正向
优化奖励，只负责 invalid/uncertain 标记。

### 8.5 为什么不立即训练 PPO

当前没有足够真实干预数据。推荐顺序是：

1. 规则/统计 Agent shadow mode；
2. 上下文 bandit 或学习排序器选择 factor/operator；
3. 用 solver/Oracle 结果积累离线 transition；
4. advantage-weighted BC；
5. 最后再训练短时程 masked PPO。

这样可先获得稳定、可解释、可回退的提升，再让 RL 学习多步“试探—扩域—回退”。

## 9. 第二阶段实施包

### S2.0：冻结第一阶段输入

- 固定当前语义 Artifact、题库版本和开放提案；
- 标记下界 shaping 总量为 `formula_equivalence_pending`；
- 不再继续扩题库，除非新项目出现明确未覆盖机制。

### S2.1：甘特图计算器与项目 Adapter

- 优先实现 high/medium 次级目标计算器；
- 输出每项的数值、作用域、贡献最大的资源/工序和可追踪实体；
- 补最小独立 Validator，作为验收门而非主优化任务；
- 将目标项目现有求解器封装为 Generator/Oracle Tool。

### S2.2：Factor/Interaction 发现

- 从每个异常次级目标追踪上游决策、状态和结构；
- 形成 `factor → mechanism → makespan` 假设；
- 建立候选变化、控制和传播闭包；
- LLM提案只进入 shadow queue。

### S2.3：算子与局部求解

- 为每个 Factor 绑定1–3个合法算子；
- 闭包外冻结；
- 同时运行确定性邻域与 bounded CP-SAT；
- 保存失败候选和传播范围。

### S2.4：配对干预与后验

- 同算子随机位置、同位置随机算子、同闭包规模随机区域；
- 测量 `Δsecondary_target`、`Δmakespan`、有效率与成本；
- 按 problem family/instance scale/factor/operator/closure 建立后验。

### S2.5：Agent shadow mode

- Agent只给排序和动作建议，不控制 incumbent；
- 与固定VNS、ALNS、solver-only比较；
- 评价 improvement/Oracle-call、time-to-first-improvement 和失败率。

### S2.6：Agentic RL

- 用已验证 transition 做 BC；
- 再训练短时程 masked PPO；
- 始终保留规则 fallback、tabu、预算门和严格回退。

## 10. 第二阶段完成标准

至少满足以下条件才可称为第二阶段完成：

- high/medium 指标在真实 Problem/Schedule IR 上可计算；
- 至少一个问题族形成绑定实体的 Factor 候选库；
- 每个 Factor 有合法算子和传播闭包；
- 同一 incumbent 上存在配对干预与完整 Oracle 记录；
- 至少一个 Factor/Operator 在未见实例上表现出稳定正效应；
- Agent shadow mode 优于固定基线，报告置信区间和成本；
- 若训练 RL，必须存在 dataset hash、配置、checkpoint 和 held-out 指标；
- 所有接受的排程可重放、可验证，失败候选也保存。

## 11. 第二阶段第一个建议实验

以当前 FJSP 为 pilot：

1. 读取一批实例和稳定 incumbent；
2. 计算最大机器负荷、工序间等待、机器空闲、关键资源内部空闲；
3. 找到 makespan 尾端关键机器块和最大等待传播点；
4. 提出机器重分配、相邻交换/插入、闭包释放和 non-delay mask 放宽四类干预；
5. 对每个点运行固定预算局部 CP-SAT；
6. 用原项目环境/求解器和独立静态 Validator 验证；
7. 保存 `factor × operator × closure × Δmetrics × Δmakespan × cost`；
8. 先训练/拟合一个上下文排序器，再决定是否启动 PPO。

这一步直接服务甘特图优化，不以代码重构作为前置研究目标。
