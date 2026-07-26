# FJSP-DAN 代码优先标签审计

> **所属模块**：A — 项目输入与证据  
> **交叉分类**：A × B — 代码/论文证据与语义标签  
> **文档职责**：记录论文标签与代码事实的逐项核验结果。

> 审计日期：2026-07-26  
> 固定代码提交：`2cf81b13f5044451e78cf780f8fb3e7eeac054c1`  
> 论文：Wang et al., *Flexible Job Shop Scheduling via Dual Attention
> Network-Based Reinforcement Learning*, arXiv:2305.05119v2 / IEEE TNNLS  
> 原则：代码决定当前项目实际语义；论文只用于检查标签，不覆盖代码。

## 结论

本次 Opus 输出的主问题族和经典 FJSP 核心标签正确，但不能整体批准为
`human_verified_label`。逐项核查后：

- 关键标签正确：FJSP、静态、确定性、离线、makespan、precedence、
  resource eligibility、exactly-one machine、machine no-overlap、工序/机器联合选择；
- 三项应修正：`multi_resource`、第二个“reward shaping 目标”、算法选择决策；
- 一项应收紧表述：归一化资格风险只有在整个传入批次完全没有任何原始 0 时触发；
- 一项应换层保存：release time=0 是代码事实和论文假设，但应作为固定假设，
  不是优化因子；
- 当前状态：`code_paper_audited_pending_human_review`。

## 代码—论文逐项对照

| 项目 | 论文 | 固定提交代码 | 审计 |
|---|---|---|---|
| 问题族 | 标准 FJSP | 工序有候选机器，动作联合选择 job/machine | 正确 |
| 作业到达 | 所有作业在 `Ts=0` 同时到达 | job/machine free time 全部初始化为 0 | 事实正确；固定假设 |
| 工序优先 | c1 | candidate 指针与 `max(job_free,machine_free)` | 正确 |
| 机器资格 | c2 的 compatible set | 原始 `op_pt=0` 表示不兼容；环境生成 mask | 正确 |
| 唯一机器 | c2：exactly one compatible machine | 环境每工序只消费一次；CP-SAT `AddExactlyOne` | 正确 |
| 机器容量 | c3：每机同一时刻至多一道工序 | `mch_free_time`；CP-SAT `AddNoOverlap` | 正确 |
| 目标 | 单一 makespan | `current_makespan`；CP-SAT Minimize(makespan) | 正确 |
| reward shaping | makespan 下界差分，累计回报等价于 makespan | `max_endTime - max(op_ct_lb)` | 不是第二目标 |
| 行动 | compatible operation-machine pair 立即在决策时刻开工 | dynamic pair mask 只保留最早可立即开工的 pair | 正确的 MDP 限制；不是标准 FJSP 硬约束 |
| start time | 论文问题描述中是排程结果；MDP 动作不单独选择 | 由 `max(candidate_free_time,mch_free_time)` 派生 | 代码优先：不可直接修改 |
| 资源类别 | 机器 | 只有机器一种资源类别 | 不应标 `multi_resource` |
| 求解方法选择 | 实验比较 | DRL/PDR/CP-SAT 入口 | evaluation/algorithm choice，不是排程决策 |
| setup/transport/no-wait/maintenance | 仅相关工作或未来扩展 | 主链没有实现 | 正确排除 |

## 归一化资格风险的准确表述

代码执行顺序为：

```text
pt_lower_bound = min(整个批次 op_pt)
normalized = (op_pt - pt_lower_bound) / range
process_relation = normalized != 0
```

因此：

- 只要整个批次存在任意一个原始 0，`pt_lower_bound=0`，原始正加工时间不会因
  归一化变成 0；
- 只有整个传入批次所有工序对所有机器都可加工、完全没有原始 0 时，
  `pt_lower_bound>0`，全局最短加工时间项会被错误标为不兼容；
- `SD2 mix` 允许生成完全柔性工序，但单个完全柔性工序不足以触发问题；
- 是否存在“整个 batch 全完全柔性”的实际配置或数据文件，需要定向实例测试，
  不能只由生成器可能性宣布已发生。

这属于代码实现风险，不由论文是否提到决定。

## 当前项目图的审批边界

项目语义图中的节点和引用已经完成程序证据审计；本文件进一步完成代码—论文标签
对照。但当前 FJSP 测试没有把一个具体实例转换为统一 IR，所以图中没有展开每个
Job/Operation/Mode/Resource 的实例级析取结构。

因此当前已经核查的是：

```text
项目类型 + 环境 + 目标 + 约束 + 决策 + Oracle + 代码证据
```

尚未核查的是：

```text
某个具体 FJSP 实例的完整 operation-level constraint graph
```

后者需要选择实例，通过 importer 转换到 `Problem` IR，再生成 operation-level 图并
与代码环境回放结果对照。
