# FJSP Opus 5 最终综合与次级目标复核

## 说明

源文件：

```text
outputs/fjsp_test/fjsp_opus5_staged_semantics.json
SHA-256: 4ddd11ccc729ef0f8fb43a0ee1a43df67378f6a0d9920c47e285b670c8a2e0f8
```

旧 Critic Schema 没有 `secondary_targets` 字段。因此下面“最终综合”是上次 CC
直接输出；“次级目标”是从 Critic 的逐约束 rationale 中显式重建，不冒充原始字段。

## 上次 CC 的最终综合

> 该仓库是论文《Flexible Job Shop Scheduling via Dual Attention Network Based
> Reinforcement Learning》的官方实现 FJSP-DRL。它把经典柔性作业车间调度
> （FJSP）建模为逐步构造式 MDP：实例以 `job_length[J]` 与 `op_pt[N,M]` 表示，
> `op_pt=0` 表示机器不可加工；环境每步接受一个 `(job,machine)` 动作，将该 job
> 的当前候选工序派到兼容机器，开工时间由
> `max(job_ready_time,machine_free_time)` 派生；N 步后完成。目标为最小化
> makespan，训练奖励为完工下界差分 shaping。相对经典 FJSP，项目使用近似
> non-delay 动作收窄；环境机器资格与 OR-Tools 的解释存在需要核查的一致性风险。
> 主链未见 setup、运输、no-wait/blocking、维护或工人双资源；CP-SAT 只作为带
> 时限参照模型，仓库没有独立最终排程可行性校验器。

代码—论文复核后的修正：

- 删除 `multi_resource`；
- reward shaping 不是第二优化目标；
- 求解算法选择不是调度决策；
- 资格归一化风险只在整个输入 batch 没有任何原始 0 时触发。

## 从上次 CC 结果重建出的次级目标

本次复核把“机器分配质量”和“同机排序质量”移出了次级目标。两者是 Agent/
优化器可操作的**决策维度**，但没有天然的标量定义。用宽泛的“质量”会让模型在
不同批次中暗自更换含义，无法做受控实验。

| 次级目标 | 期望方向 | 与 makespan 的关系 | 当前状态 |
|---|---|---|---|
| 机器选择加工时间增量 | 降低 | 已选机器相对最短候选加工时间增加的总时长 | 可直接计算；不能单独优化 |
| 最大机器工作负荷 | 降低 | 机器分配形成的最大累计加工时间 | 可直接计算 |
| 低柔性工序机器负荷 | 降低 | 候选机器少的工序在机器上的加权负荷 | 需要实验校准 |
| 关键资源内部空闲时间 | 降低 | 关键资源相邻工序间的正空闲时间总和 | IR 已有计算器 |
| 工序就绪后等待时间 | 降低 | 工序就绪到实际开工的等待总和 | IR 已有计算器 |
| 受限解码规则完工期差值 | 降低 | 受限解码与完整可行插入解码的配对 makespan 差 | 需要消融 |

实现析取图关键路径长度保留为主目标分解工具。它在本场景通常等于 makespan，
因此不重复作为次级指标。

## 单独的诊断维度

| 诊断项 | 用途 | 为什么不属于次级目标 |
|---|---|---|
| 机器资格一致性 | 检查环境、实例、Oracle 语义一致 | 是可行性/解析前提，不能靠“最小化”改善 |
| 完工下界可靠性 | 判断 reward shaping 信息质量 | 是测量工具质量，不是生产日程指标 |
| Oracle 参照有效性 | 核对求解状态、时限与可行性 | 是实验可信度条件 |

完整结构化版本：

- `fjsp_opus5_secondary_targets_reconstructed.json`

## 不应作为次级优化目标

- “每道工序恰好执行一次”：所有可行候选都必须满足，是可行性守卫；
- “release time=0”：所有候选共享的固定退化假设；
- “选择 DRL、PDR 或 CP-SAT”：实验层算法选择；
- “reward shaping”：训练信号设计，不是独立生产目标。

## 下一步不是继续让 LLM 排名

对每个次级目标继续递归：

```text
次级目标
→ 候选 Factor
→ Factor 交互
→ 可控 Decision / Policy Rule
→ 局部受控候选
→ Oracle
→ 次级目标变化
→ makespan 变化
→ 贝叶斯后验
```

因此当前六个次级指标只是可测量入口，三个诊断项只负责可信度守门；二者都不是
已经验证的真实原因清单。
