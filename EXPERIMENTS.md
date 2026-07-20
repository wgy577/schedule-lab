# Schedule Lab 实验记录

> 本文件记录“做过什么、为什么做、是否可接受”。README 只展示当前结论，详细实验数据保存在 `outputs/`。

## 当前正式基线

| 项目 | 当前值 |
|---|---|
| 舰载机 incumbent | 627.8 s |
| 原始 greedy 基线 | 675.5 s |
| 累计改善 | 47.7 s / 7.06% |
| 正式方案 | `outputs/carrier_alns_best_iter3_gap6_closed_630_5.json` |
| 验收要求 | 通用验证 + 原领域轨迹/碰撞回放 + 确定性复跑 |

## 已接受实验

| 日期 | ID | 输入 | 方法 | 释放/变化范围 | 真实结果 | 结论 |
|---|---|---:|---|---|---:|---|
| 2026-07-17 | CARRIER-BASELINE | — | greedy baseline | 全局构造 | 675.5 s | `ACCEPTED` 基线 |
| 2026-07-17 | CARRIER-CONTROLLED-SEARCH | 675.5 s | 固定 seed 受控策略搜索 | 少量策略决策 | 637.5 s | `ACCEPTED` |
| 2026-07-17 | CARRIER-ALNS-01 | 637.5 s | O4 相邻顺序调整 + Oracle 闭包 | 8 架传播任务 | 636.2 s | `ACCEPTED` |
| 2026-07-18 | CARRIER-ALNS-02 | 636.2 s | O4 相邻顺序调整 + Oracle 闭包 | 7 架传播任务 | 630.5 s | `ACCEPTED` |
| 2026-07-18 | CARRIER-ALNS-03 | 630.5 s | O4 相邻顺序调整 + Oracle 闭包 | 1 架传播任务 | **627.8 s** | `ACCEPTED` 当前 incumbent |

## 已拒绝或仅供诊断的实验

| 日期 | ID | 输入 | 方法 | 抽象结果 | 领域结果/原因 | 结论 |
|---|---|---:|---|---:|---|---|
| 2026-07-19 | FIXED-ROUTE-SPACETIME-01 | 627.8 s | 固定 MAT 路线、0.1 s 时空 CP-SAT | 619.5 s | 803.9 s；40 个机器/路线绑定变化 | `REJECTED` |
| 2026-07-19 | ROUTE-BINDING-MASTER-01 | 627.8 s | 小型路线绑定主问题 | 634.6 s | 已不优于 incumbent，未进入昂贵 Oracle | `REJECTED` |

## 通用问题族回归

| 日期 | 回归 | JSP | FSP | FJSP | HFSP | 状态 |
|---|---|---:|---:|---:|---:|---|
| 2026-07-19 | LPT → adaptive improvement | 14 → 11 | 22 → 19 | 11 → 7 | 17 → 17 | 25 项测试通过 |

## 计划实验

| 优先级 | ID | 假设 | 方法 | 验收条件 | 状态 |
|---|---|---|---|---|---|
| P0 | CARRIER-CAUSAL-CLOSURE | 当前关键空档可由更小因果责任链压缩 | causal closure fix-and-optimize | 领域真实 makespan < 627.8 s | `PLANNED` |
| P0 | CARRIER-LOCAL-BRANCHING | 小 Hamming 半径可联合调整绑定与顺序 | local branching + CP-SAT + Oracle | 完整验证且冻结区不变 | `PLANNED` |
| P1 | CARRIER-SHIFTING-BOTTLENECK | 单通道与上游准备资源存在可协调阻塞 | shifting bottleneck repair | 真实目标改善且可复现 | `PLANNED` |
| P1 | MULTIFAMILY-BENCHMARK | 策略路由应推广到更多规模与约束组合 | 多实例 JSP/FSP/FJSP/HFSP 回归 | 不恶化率、改善率、成本均有记录 | `PLANNED` |
| P2 | OPTIONAL-TRAJECTORY-TOOL | 轨迹预检查可减少无效 Oracle 调用 | 可插拔路线/冲突 Tool | 关闭 Tool 时通用测试仍通过 | `PLANNED` |

## 新实验登记模板

复制下面模板，完成后将状态改为 `ACCEPTED`、`REJECTED` 或 `PROVISIONAL`。

```text
实验 ID：
日期：
问题族 / 实例：
Incumbent 路径与 SHA-256：
Incumbent 目标：

诊断证据：
因果假设：Observed bottleneck → responsible operations → released closure
方法与参数：
固定 seed / worker / budget：
Released decisions：
Frozen decisions：

通用验证：
领域 Oracle：
复跑一致性：
候选目标：
变化成本与鲁棒性：

结论：ACCEPTED / REJECTED / PROVISIONAL
原因：
候选路径：
审计产物：
复现命令：
```

## 维护规则

- 不删除失败实验；失败是 Cut、后验和方法降权的重要证据。
- 不以文件名或求解器代理目标代替真实 `max(operation.end)`。
- `PROVISIONAL` 结果不得写入 README 顶部正式成绩。
- 同一实验重跑时保留相同 ID，并记录新的 replay/hash，而不是覆盖证据。
- 大型中间缓存不提交；最终展示视频必须同时保留 manifest。
