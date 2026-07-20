# Schedule Lab

一个面向 Agent 的通用混合调度优化底座。Agent 负责发现瓶颈和选择优化策略；确定性的求解器、启发式算法和验证器负责产生并认证调度结果。

面向合作者的现有能力、成熟度分级和后续路线见 [`SCHEDULE_LAB_PLAN.md`](SCHEDULE_LAB_PLAN.md)。

当前第一版支持：

- JSP、FSP、FJSP、HFSP 的统一问题表示和适配器；
- 多模式工序、可选机器、多资源占用、资源容量、任意前序和跨资源族绑定；
- 5 种派工启发式组成的初始解组合；
- PyJobShop 高层约束求解后端；
- OR-Tools CP-SAT 精确/限时求解、warm start 和调度稳定性代价；
- 硬约束验证、真实 makespan、流动时间、延误、利用率、空闲和变更成本；
- 当前 20 架舰载机 `deck_update` 调度的只读导入与指标审计；
- 旧 PPO + 原始轨迹/碰撞环境的隔离式领域 Oracle 与多候选搜索；
- 不依赖单一邻域的多方法 incumbent-improvement 工作流；
- 固定规格、带输入哈希审计的调度对比视频模板；
- 项目内 `improve-schedules-with-oracles` Skill；
- 甘特图导出、CLI 和 MCP Server。

## 自适应短名单与调度—轨迹联合优化

默认控制器现在采用 `fast → balanced` 的一次性自适应升级：fast 找到严格改善或领域 Oracle 候选就立即停止；只有失败时才进入 balanced，而且不会重复验证 fast 已经跑过的邻域。固定 LPT 回归结果为 JSP `14 → 11`、FSP `22 → 19`、FJSP `11 → 7`，HFSP 在当前边界内保持 `17`；完整结果见 `outputs/adaptive_multifamily_improvement_benchmark.json`。

当前舰载机代码中的 220 条 MAT 轨迹已经被规范化为可审计路径列：100 条牵引车初始移动、60 条牵引车—飞机联合牵引、60 条后续转运。每个精确 job/machine/phase 绑定只有一条路径，但按 job/phase 汇总的 60 个路由选择组全部具有 3～5 条机器/准备位绑定路线。现阶段的问题不是“完全没有备选路线”，而是路线只通过环境内部状态被间接选择，主问题还不能显式控制并解释 route-column compatibility。

第一轮固定路线时空 CP-SAT 已加入牵引车 O1–O4 锁定、弹射位 O5–O8 锁定、全局弹射通道 10 秒冷却和 0.1 秒半开区间空间占用。抽象模型给出 `619.5s` 的最优候选，但旧领域环境两次确定性复跑均得到 `803.9s`，并出现 40 个机器绑定变化，因此候选已拒绝，`627.8s` incumbent 保持不变。失败原因已保存为精确上下文 reachability Cut，下一轮只释放受影响的 8 个零基 job，而不是扩大到全局重排。

```bash
schedule_lab/.venv/bin/python schedule_lab/run_schedule_lab.py \
  adaptive-improvement-benchmark --baseline-rule lpt \
  --output schedule_lab/outputs/adaptive_multifamily_improvement_benchmark.json

schedule_lab/.venv/bin/python schedule_lab/run_schedule_lab.py \
  carrier-route-catalog --legacy-root . --max-points 24 \
  --output schedule_lab/outputs/carrier_route_catalog.json

schedule_lab/.venv/bin/python schedule_lab/run_schedule_lab.py \
  carrier-joint-schedule-trajectory-plan --legacy-root . \
  --schedule schedule_lab/outputs/carrier_alns_best_iter3_gap6_closed_630_5.json \
  --validated-experiment-count 62 \
  --output schedule_lab/outputs/carrier_joint_schedule_trajectory_plan.json

schedule_lab/.venv/bin/python schedule_lab/run_schedule_lab.py \
  carrier-fixed-route-spacetime \
  --schedule schedule_lab/outputs/carrier_alns_best_iter3_gap6_closed_630_5.json \
  --legacy-root . --output schedule_lab/outputs/carrier_fixed_route_spacetime.json
```

联合架构采用 CP-SAT/MILP 调度主问题与无冲突轨迹子问题，通过路线不可行、冲突先后、最小间隔、真实旅行时间和车辆连续性 Cut 循环收敛。Agentic RL 只选择瓶颈、邻域、方法、路径列和求解预算，不直接生成未经约束的开始时间，也不负责宣布可行。当前仅有 62 条去重 Oracle 证据，暂不训练 encoder；达到至少 500 条后再评估“异构调度图 encoder + 路径时空图 encoder + cross-attention”。详细契约与可证明最优性的边界见 `skills/improve-schedules-with-oracles/references/agentic-rl-and-joint-trajectories.md`。

## Harness 闭环

```text
问题适配器 → 统一 Scheduling IR → 指标/瓶颈诊断
                                      ↓
人工或 Agent 选择邻域算子 → 启发式 / PyJobShop / CP-SAT 候选组合
                                      ↓
                         通用硬约束验证器
                                      ↓
                舰载机任务再进入轨迹/碰撞领域 Oracle
                                      ↓
                    可行且更优？— 是 → 人工审核/导出
                              └— 否 → 诊断反馈并换邻域
```

LLM/Agent 在这里负责解释瓶颈、选择算子和组织实验，不负责宣告一个方案“可行”。可行性由确定性验证器给出；甲板任务还要经过旧碰撞环境的第二道验证。这样，JSP/FSP/FJSP/HFSP 共用上半段，航母特有约束只作为可插拔领域 Oracle 加在末端。

## 多方法改进工作流

邻域搜索只是控制器中的一层。新的工作流始终从已有 incumbent 出发，并按成本与改动范围逐层升级：

```text
固定模式/顺序的时序压缩
  → 问题族启发式或 shifting-bottleneck 诊断
  → 因果闭包 / local-branching fix-and-optimize
  → tabu 去重 → 通用验证 → 领域 Oracle
  → 接受并重新诊断

平台期
  → rolling horizon / relax-and-fix
  → 有界确定性 ALNS
  → 两个已验证 elite 之间的 path relinking
  → 有足够 Oracle 历史后用贝叶斯后验分配实验预算
```

其中“因果闭包修复”不再按甘特图位置机械选择半径 2～4，而是从瓶颈沿前序、资源顺序、路径绑定、时间窗、车辆连续性和碰撞传播关系反向追踪，只释放造成瓶颈的最小传播闭包。这样既保留了局部优化的稳定性，也能推广到 JSP、FSP、FJSP、HFSP。

生成一个可审计的多方法计划：

```bash
.venv/bin/schedule-lab improvement-workflow \
  problem.json incumbent.json \
  --evidence-count 12 \
  --validated-elite-count 2 \
  --output outputs/improvement_workflow.json
```

对四类回归实例一次生成完整方法计划：

```bash
.venv/bin/schedule-lab improvement-workflow-benchmark
```

输出位于 `outputs/multifamily_improvement_workflows.json`。

项目内 Skill 位于 `skills/improve-schedules-with-oracles/`，方法选择表位于其 `references/method-selection.md`。
邻域以外的分解、路径重连、对偶引导、Decision Diagram、Logic-Based Benders、Oracle 切割和鲁棒优化方案位于 `skills/improve-schedules-with-oracles/references/advanced-optimization-portfolio.md`。

## 通用邻域控制器

通用控制器不包含“工序 7”“弹射器”或任何甲板坐标。它根据问题族选择诊断面：JSP 使用关键资源块，FSP 使用排列与末级空档，FJSP 使用关键资源块和合法替代机器，HFSP 使用阶段饥饿、并行机负载与末级空档。每个候选明确列出 released/frozen 工序，CP-SAT 固定邻域外开始时间和模式；单邻域平台后才按稳定排名组合两个瓶颈，并限制释放比例。领域约束通过插件 Oracle 追加。

```bash
.venv/bin/schedule-lab generic-neighborhood-benchmark
```

任意外部问题先通过适配器导出 canonical JSON，再使用相同入口：

```bash
.venv/bin/schedule-lab generic-neighborhood-plan problem.json incumbent.json
.venv/bin/schedule-lab generic-neighborhood-repair \
  problem.json incumbent.json outputs/generic_neighborhood_plan.json --index 0
```

领域问题的 repair 结果会标记为 `provisional`，必须接入该项目自己的 Oracle 后才能接受。

多实例历史可以形成跨问题族弱先验，再由每个问题族自己的观测更新后验：

```bash
.venv/bin/schedule-lab generic-evidence-rank \
  --history outputs/generic_neighborhood_benchmark.json
```

跨族信息只影响下一次实验排序；JSP 的后验不会替 FJSP 或领域 Oracle 宣告可行。

固定回归用确定性的 LPT 方案作为待优化 incumbent，覆盖 JSP、FSP、FJSP、HFSP，并检查邻域外赋值逐项不变。当前小样例中，JSP 由 `14 → 11`，FSP 由 `22 → 19`，FJSP 由 `11 → 7`；HFSP `17` 在当前有界邻域内保持不变。结果位于 `outputs/generic_neighborhood_benchmark.json` 和 `outputs/generic_neighborhood_gantt/`。

## 安装

```bash
cd "/Users/guangyuwu/Desktop/sortie code/comparision/schedule_lab"
python3 -m venv .venv
.venv/bin/pip install .
```

## 运行四类调度回归实验

```bash
.venv/bin/schedule-lab benchmark
```

结果写入：

- `outputs/benchmark.json`
- `outputs/gantt/jsp.png`
- `outputs/gantt/fsp.png`
- `outputs/gantt/fjsp.png`
- `outputs/gantt/hfsp.png`

## 审计当前舰载机调度

```bash
.venv/bin/schedule-lab carrier-audit
```

该命令不会改变现有调度。它会重新按最晚工序结束时间计算真实 makespan，并输出 `outputs/carrier_audit.json` 与 `outputs/carrier_baseline.png`。

当前导入的舰载机方案来自碰撞感知的旧环境，因此基线标记为 `domain_validated=true`。任何新生成的舰载机候选方案必须重新经过 MATLAB 轨迹/碰撞环境回放，不能只凭通用 CP-SAT 验证结果进入 3D Demo。

## 搜索更好的甲板调度候选

```bash
.venv/bin/schedule-lab carrier-search --rollouts 16 --seed 0
```

这个命令在独立子进程中加载当前 `FJSP_J20M12h/100_502` 网络。第一个候选复现 greedy 基线，其余候选采用受控策略邻域：大多数决策保持 greedy，只在少量决策点按低温分布探索，避免全程随机采样破坏已有结构。每个候选都由旧 `FJSP_Env + connector_two` 完整生成，因此沿用现有机器掩码、甲板轨迹、资源与碰撞延迟逻辑。每轮都会重新加载模型，保留训练态 BatchNorm 的同时避免候选之间发生状态漂移。

候选按照 `max(operation.end)` 计算的真实 makespan 排序，不再使用旧环境误取的最大开始时间。系统随后用统一 IR 验证器做第二次资源容量、前序、机器选择和路线绑定检查。结果写入：

- `outputs/carrier_search.json`：全部候选摘要、双重验证与基线对比；
- `outputs/carrier_best_schedule.json`：与现有 Demo 调度结构兼容的最佳候选；
- `outputs/carrier_best_schedule.png`：最佳候选甘特图。

搜索结果不会自动覆盖 3D Demo；人工审核通过后才能显式接入。

## 构造确定性工序 7 邻域

```bash
.venv/bin/schedule-lab carrier-vns-plan \
  --schedule /path/to/incumbent.json \
  --radii 2 3 4 \
  --max-gaps 5
```

该命令不会生成随机候选，也不会改写现有调度。它按照工序 7 单通道的正空档从大到小排序，以固定的半径 `2 → 3 → 4` 构造 VNS 局部邻域，冻结邻域外的机器选择，并为每个邻域生成稳定哈希供禁忌表去重。结果默认写入 `outputs/carrier_vns_plan.json`，作为后续局部 CP-SAT 修复与领域 Oracle 回放的确定性输入。

对邻域做一轮确定性的领域回放：

```bash
.venv/bin/schedule-lab carrier-vns-search \
  --schedule /path/to/incumbent.json \
  --radii 2 3 4 \
  --max-gaps 1 \
  --seed 0
```

该命令只在同一工序阶段内改变邻域飞机的优先顺序，其他飞机仍按原网络贪心决策；任何导致邻域外机器选择变化的结果都会被拒绝。每个候选由原 `FJSP_Env + connector_two` 重新生成真实路径时长与碰撞等待，并再次通过统一验证器。当前已接入固定既有机器模式的局部 CP-SAT 提案器；它只输出各工序的候选优先顺序，真实路径时间、动态机器合法性和碰撞等待仍以领域环境回放为准。

2026-07-17 的第一轮确定性回归从 `675.5` 基线出发。最大 O7 空档的半径 `2 → 4` 回放暴露出邻域外资源变化，半径 5 达到传播闭包；其中 `backward-insertion` 得到真实 makespan `657.8`，缩短 `17.7`（`2.62%`），邻域外机器变化为 0，完整 160 工序通过统一验证。相同命令重复运行得到相同调度哈希 `b9860c37...`。该结果用于验证确定性闭环，尚未覆盖此前随机邻域找到的 `637.5` 候选，也未接入最终 Demo。

同日的局部 CP-SAT 回归进一步验证了双层判定的必要性：半径 5 的抽象模型给出 `532.2` 秒代理目标，但领域回放为 `671.4` 秒，并检测到邻域外舰载机 15 的模式传播，因此拒绝；按证据扩展到半径 6 后传播闭合，但领域 makespan 恶化为 `849.4` 秒，仍拒绝。CP-SAT 由此保留为受控候选生成器，而不是可行性 Oracle。随后从 `657.8` 解继续确定性检查前四个 O7 空档，没有发现更优闭合候选，当前局部进入平台期，下一阶段才考虑加入带固定 destroy/repair 顺序的 ALNS。

## 确定性 ALNS 与贝叶斯证据排序

ALNS 从可复现的 incumbent 决策轨迹出发，不再从网络 greedy 重新调度。先确认 160 步决策能在领域环境中生成完全相同的 makespan 与调度哈希，再对关键空档附近的同工序派工顺序做一次相邻交换。开放传播边界的候选只用于诊断；必须按 `requiredExpansionJobs` 扩展并重新回放到传播闭合，才可能接受。

```bash
.venv/bin/schedule-lab carrier-alns-search \
  --schedule outputs/carrier_best_schedule_traced.json \
  --destroy-sizes 1 --max-gaps 8 \
  --operators alns-adjacent-o4
```

2026-07-17 至 2026-07-18 的确定性迭代结果：

- `637.5 → 636.2`：第四大 O7 空档附近交换一次 O4 顺序，按 Oracle 证据扩展 8 架传播任务后闭合；
- `636.2 → 630.5`：新基线第三大空档附近交换一次 O4 顺序，扩展 7 架传播任务后闭合；
- `630.5 → 627.8`：新基线第六大空档附近交换一次 O4 顺序，只需扩展 1 架传播任务即闭合；
- 三个接受结果均完整复跑两次，调度与 160 步决策轨迹逐项一致；
- `627.8` 相对原 greedy `675.5` 缩短 `47.7`（`7.06%`），相对固定种子采样候选 `637.5` 再缩短 `9.7`（`1.52%`）。

统计层只负责选择下一次实验，不负责验收。它对去重后的 Oracle 记录使用 Beta-Binomial 后验估计原始改善率、低成本闭包率和接受率，并对正收益做收缩估计；Bayesian-UCB 给未测试的相邻算子有限探索预算。工序 3 的 4 次探索全部恶化后会立即被后验降权。

```bash
.venv/bin/schedule-lab carrier-evidence-rank \
  --history outputs/carrier_alns_search*.json \
  --output outputs/carrier_operator_posterior.json
```

当前最优实验产物：

- `outputs/carrier_alns_best_iter3_gap6_closed_630_5.json`
- `outputs/carrier_alns_best_iter3_gap6_closed_630_5.png`
- `outputs/carrier_operator_posterior.json`

## 对比视频工作流

全局 greedy 基线保存在 `outputs/carrier_greedy_baseline_675_5.json`（true makespan `675.5`）。当前增量优化视频比较的是上一轮已验证 incumbent `outputs/carrier_best_schedule_traced.json`（`637.5`）与当前候选 `outputs/carrier_alns_best_iter3_gap6_closed_630_5.json`（`627.8`）。

```bash
python3 workflows/video/render_schedule_comparison.py
```

模板固定输出 `1920×816`、`30 fps`、`60 s`、H.264/yuv420p，左右使用同一个 `960×816` 面板和同一条共享时钟。两侧甘特横轴固定到较大的 makespan；短方案先完成后保持最终状态，等待长方案结束，绝不分别归一化到 60 秒。生成前会核对规范化调度 SHA-256、工序集合、机器绑定和时序差异；左右相同会直接拒绝。视频旁的 `.manifest.json` 保存本次输入、差异与共享时间轴审计。详细参数见 `workflows/video/COMPARISON_VIDEO_TEMPLATE.md`。

2026-07-17 的固定回归实验（64 个候选，seed=100）中，当前 greedy 基线的真实 makespan 为 `675.5`，受控策略邻域找到的最佳候选为 `637.5`，缩短 `38.0`（`5.63%`）。该候选同时通过旧碰撞环境和统一 IR 验证器；它目前仍是实验候选，并未写入最终 Demo。

## MCP Server

```bash
.venv/bin/schedule-lab-mcp
```

提供的工具：

- `scheduling_capabilities`
- `analyze_schedule`
- `solve_problem`
- `compare_schedules`
- `plan_schedule_improvement`
- `audit_current_carrier`
- `search_current_carrier`

MCP是 Agent 与求解器之间的接口，不替代优化器。所有候选方案必须经过 `validate_schedule`；舰载机候选还必须通过领域轨迹与碰撞验证器。

## 下一阶段

1. 在 `627.8` incumbent 上比较因果闭包、local branching 与 shifting-bottleneck，而不是继续只扩大半径；
2. 实现 assignment/sequence 交替修复与对偶价格引导的释放排序；
3. 用两个完整验证的 elite 测试路径重连，并保持每个中间点可修复、可回放；
4. 将稳健性场景、最小 slack 和 Oracle 成本加入分层贝叶斯实验分配；
5. 人工审核后再决定是否把最优可行 JSON 接入视频和 3D 航母回放。
