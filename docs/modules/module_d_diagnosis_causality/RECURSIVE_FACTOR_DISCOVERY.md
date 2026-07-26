# 从次级目标到真实影响因素的递归发现链

> **所属模块**：D — 诊断与因果机制  
> **交叉分类**：D × E × F × H — 影响因素、干预候选、Oracle 与后验  
> **文档职责**：定义可测量次级指标、候选因素、交互和受控验证链。

## 两个阶段的固定边界

本项目必须把以下两个问题分开：

```text
阶段 A：当前项目应该测量哪些次级目标？
阶段 B：哪些具体因素和交互真正改变这些次级目标？
```

阶段 A 使用规范题库和小批选择题：程序先按问题族、变体、机制、决策、角色和证据
视图召回，LLM 只在缩小后的候选中做语义匹配。该阶段最多输出
`Project-level Secondary Targets`，不能输出真实影响因素或因果结论。

阶段 B 必须重新从代码和 IR 中寻找与已选目标相连的参数、状态、结构、策略规则、
外部事件与交互，并绑定具体 symbol/实体。候选 Factor 还要经过变化、可控性、传播、
受控干预、Oracle、主目标和统计后验门，才可能升级为有作用域的
`Validated Influence Factor`。

未知指标可以通过 `unresolved_new_candidate` 返回候选扩充流程；未知因素不能因为不在
固定枚举中被丢弃，因此 Factor 层继续采用宽类型加明确实体，而不是复用次级目标题库。

## 目标

主目标通常是 makespan，但它不能直接告诉 Agent 应修改哪里。项目采用递归链：

```text
主目标
→ 次级目标 / 中间机制
→ 影响该机制的候选因素
→ 因素之间的交互
→ 可控决策或可修改策略
→ 受控候选
→ Oracle 回放
→ 后验更新
→ 继续深挖或停止
```

每一层都可以再次成为下一层的“待解释目标”。例如：

```text
makespan
└── 关键机器空闲间隙
    ├── 机器分配
    │   ├── 候选机器加工时间差
    │   ├── 当前机器负载
    │   └── 与后续关键工序的交互
    ├── 同机工序顺序
    │   ├── 前驱就绪时间
    │   ├── 插入位置
    │   └── 相邻工序组合
    └── non-delay 动作限制
        ├── 最早决策时刻
        ├── 被屏蔽的主动等待动作
        └── 尾部追加策略
```

## 六类节点

| 节点 | 含义 | 是否可直接优化 |
|---|---|---|
| PrimaryObjective | makespan、tardiness 等最终目标 | 只用于验收 |
| SecondaryTarget | 等待、空闲、关键路径、负载失衡等 | 用于诊断和候选排序 |
| DiagnosticTarget | 资格一致性、下界/Oracle/Validator 可靠性 | 只负责可信度守门 |
| Factor | 参数、状态、结构或规则 | 需要先确认可控性 |
| Interaction | 两个或多个因素共同作用 | 需要局部联合实验 |
| Decision | 分配、排序、插入、路由等可修改量 | 可以产生候选 |
| Evidence/Oracle | 代码证据和验证结果 | 不直接优化 |

## 当前次级指标选项库

当前 active 固定选项覆盖：

- 关键资源内部空闲时间（time）；
- 工序就绪后等待时间（time）；
- 资源利用率变异系数（coefficient of variation）；
- 序列相关准备时间总和（time）；
- 运输引起的等待时间（time）；
- 总阻塞时间（time）；
- 资源容量空闲率（fraction）；
- 机器选择加工时间增量（time）；
- 最大机器工作负荷（time）；
- 低柔性工序机器负荷（weighted time）；
- 受限解码规则完工期差值（time）；
- other / unknown。

“机器分配”和“同机排序”属于 Decision，不属于 SecondaryTarget；“机器分配质量”、
“拥塞程度”“并行损失”“资源竞争压力”等没有统一计算口径的名字已禁止输出。每项
SecondaryTarget 必须同时给出 `measurement_definition`、`unit` 和优化方向。机器资格
一致性、目标下界可靠性和 Oracle 有效性属于 DiagnosticTarget，不能混进优化排名。

实现析取图关键路径长度在单一 makespan 场景通常与主目标数值相同，只用于关键链
定位和目标分解，不重复作为次级指标。

这套选项足够覆盖当前 FJSP-DAN 第一轮机制发现，但不能声称覆盖所有工程调度。
`other/unknown` 不是长期存储类型：它只触发人工审核、检索和新增明确分支。审核后
必须给新机制独立名称，加入知识树或机制库。

Round 4 基线有 55 个 `proposed` candidate、25 个 diagnostic 和 623 条关系；经
Round 5/6 与原 11 项兼容层扩展后，统一目录为 85/36/1056。可选 Critic 运行时已经
先做确定性召回，再由 LLM 在不超过 20 项的候选中进行高召回选择并保留置信层；不会
把 85 项全部注入模型，也不能把选中结果直接写成 Factor。

若代码揭示题库未覆盖的量，开放世界旁路允许提出 `proposed` metric 或 diagnostic。
新 metric 必须是同实例候选可变量并绑定干预把手；它仍只是阶段 A 的次级目标候选，
下一层必须重新发现 parameter/state/structure/policy/interaction 等真实影响因素。

通用扩充的资料范围、LLM 选择题、证据等级和升级门见
[次级指标与诊断项扩充研究任务书](SECONDARY_METRIC_AND_DIAGNOSTIC_EXPANSION_PLAN.md)。

## 下一层 Factor 不能只靠固定枚举

因素空间比次级目标更开放，因此采用“宽类型 + 明确实体”的组合：

```text
factor_kind:
  decision
  parameter
  state
  resource_structure
  precedence_structure
  policy_rule
  oracle_behavior
  interaction
  external_event
  unknown
```

同时必须提供：

- 对应代码 symbol / IR 节点；
- 影响哪个 SecondaryTarget；
- 直接、间接或假设关系；
- 在候选之间是否变化；
- Agent/优化器是否可控制；
- 可执行的最小受控实验；
- Oracle 和成本。

## 停止与继续规则

只有满足以下条件才继续向下深挖：

1. 次级目标能计算或能由 Adapter/Oracle 补齐；
2. 候选之间确实变化；
3. 至少存在一个直接或可验证间接的控制路径；
4. 对目标的关系可被候选实验区分；
5. 预期信息增益大于时间/Token/Oracle 成本。

否则标记为 fixed、unavailable 或 deprioritized，不让 Agent 无限递归。

## 经验校准

LLM 只提出结构化先验。真实影响关系来自：

```text
固定 incumbent 和实例
→ 改一个 factor 或小型 interaction
→ 修复传播闭包
→ Static / Light / Full Oracle
→ 记录 secondary target delta
→ 记录 primary objective delta
→ 重复和反事实对照
→ 贝叶斯更新有效概率、收益和不确定性
```

影响链最终保存为：

```text
Decision
  ──controls──> Factor
  ──interacts_with──> Factor
  ──changes──> SecondaryTarget
  ──mediates──> PrimaryObjective
  ──validated_by──> OracleResult
```
