# 数学公式实现矩阵

> 状态只描述当前代码，不代表训练效果。  
> 状态取值：`主闭环已运行`、`代码已实现但未接入`、`部分实现`、`尚未实现`。

## 公式总览

| ID | 公式/概念 | 代码状态 | 主闭环状态 | 证据 | 仍需完成 |
|---|---|---|---|---|---|
| F01 | \(A\to L\to W\to T\to D\to J\) | 部分实现 | 主闭环已运行 | `graph.py` | 当前是特征图，不是已识别 SCM |
| F02 | \(h=(D,R,P,\Omega)\) | 完整 | 主闭环已运行 | `models.py`、`cip.py` | 用真实反事实训练责任路径与闭包 |
| F03 | \(q_\phi(h\mid G,s)\) | 模型已实现 | 未接入 | `learning.py` | 训练 checkpoint 并替换规则 ranker |
| F04 | \(a=(o,\rho,u)\) | 完整动作模型 | 部分接入 | `agentic_rl.py`、`agent.py` | 主 controller 使用完整三元动作 |
| F05 | \(\pi_\psi^{mask}(a\mid s)\) | 完整 | 简化版运行 | `agent.py`、`agentic_rl.py` | 接入完整 action mask |
| F06 | \(Y^{do(h,a)}\) | 确定性修复实现 | 主闭环已运行 | `operators.py`、`repair.py`、`cp_sat.py` | 让更多算子参数影响求解器 |
| F07 | \(CE_t(h,a)=J(y_t)-J(Y^{do(h,a)})\) | 单候选差值实现 | 部分接入 | `validation.py` | 自动执行三组配对因果对照 |
| F08 | \(\Delta J_t^{full}=J(y_t)-J(y_t')\) | 完整 | 主闭环已运行 | `validation.py` | 加入多目标归一化报告 |
| F09 | \(CR_t(h)=\max_a E[\Delta J^{full}\mathbf1(Valid)]\) | 尚未严格实现 | 未接入 | 当前仅 `CIPRanker` 近似 | 同 CIP 多动作估计与置信区间 |
| F10 | \(Score=\frac{E[\Delta J]P(Valid)}{Cost+\epsilon}\) | 部分实现 | search portfolio 使用；机制后验未接入 | `posterior.py`、`search.py`、`mechanisms.py` | 接入默认 controller；校准实例/状态作用域 |
| F11 | \(R_{h,a,\ell}=\Delta^{full}+b_\ell(h,a)+\epsilon\) | 平均偏差实现 | 部分接入 | `posterior.py` | 学习条件化 fidelity bias |
| F12 | 全局收益 \(\sum_t[\alpha\Delta J+\beta\Delta J^{best}+\gamma E-\lambda Cost-\rho|\Omega|-\mu Risk-\xi Invalid]\) | 奖励函数实现 | 未接入训练闭环 | `agentic_rl.py` | 用真实 transition 训练 |
| F13 | \(\mathcal L_{CIP}\) | 完整代码 | 未训练 | `learning.py` | 数据集、权重调优、checkpoint |
| F14 | \(\mathcal L_{path}+\mathcal L_{closure}+\mathcal L_{sparse}\) | 完整代码 | 未训练 | `learning.py` | 加入闭包不足失败样本 |
| F15 | Pairwise ranking loss | 完整代码 | 未训练 | `dataset.py`、`learning.py` | 真实同 incumbent 候选对 |
| F16 | \(\mathcal L_{BC}=-\log\pi(a^*\mid s,h)\) | 完整 | smoke 已运行 | `agent.py` | 使用真实专家动作 |
| F17 | Masked PPO clipped objective | 简化实现 | 更新函数未在线调用 | `agent.py` | episode/transition/奖励接通 |
| F18 | 离线 RL | AWBC 原语实现 | 未接入 | `agentic_rl.py` | runner、IQL/CQL 对照 |
| F19 | \(\mathcal L_G=\mathcal L_{construct}+\alpha L_{repair}+\beta L_{feasible}\) | 部分实现 | 未接入 | `conditional_generator.py` | construct 与 feasible loss |
| F20 | 多候选条件生成 | Solver-backed 实现 | 未接入默认 controller | `conditional_generator.py` | neural proposals + deterministic fallback |
| F21 | 严格接受 \(Valid=1\land J(y')\prec J(y)\) | 完整 | 主闭环已运行 | `validation.py`、`controller.py` | 无 |
| F22 | 历史最好解单调性 | 完整 | 主闭环已运行 | `controller.py` | 增加 property-based 测试 |
| F23 | 多保真 Oracle 预算 | 基础实现 | 主闭环已运行 | `controller.py` | acquisition 决定预算分配 |
| F24 | 时间和 Token 成本 | 时间与机制干预 Token 字段已实现 | 部分接入 | `ExperimentRecord`、`posterior.py`、`storage/graph_store.py`、`mechanisms.py` | 主 controller/Headroom Token 入统一日志、奖励和指标 |
| F25 | 局部收敛停止 | 固定预算实现 | 主闭环已运行 | `controller.py` | 多尺度邻域穷尽与置信停止 |

## F01：共享因果骨架

\[
A\rightarrow L\rightarrow W\rightarrow T\rightarrow D\rightarrow J
\]

当前 `graph.py` 已编码到达、等待、开始、时长、资源序列、关键性、风险和成本等
特征，并构造 temporal、wait propagation、critical path 等关系。

缺口：这些边主要来自排程结构和规则，不是通过干预数据识别出的结构方程。因此
当前可称为“因果语义图”，不能称为已识别 SCM。

## F02：CIP 四元组

\[
h=(D,R,P,\Omega)
\]

已经有严格的数据模型：

- `DiagnosticPoint`；
- `ResponsiblePoint`；
- `CausalPath`；
- `CausalClosure`；
- `CausalInterventionPoint`。

`CausalPath` 会检查节点数和边数一致；闭包具有等级和邻域外风险。

## F03：核心点分布

\[
q_\phi(h\mid G,s)
\]

关系 GNN 和 MLP 已输出：

- improvement；
- validity；
- log cost；
- multi-label risk；
- rank score；
- closure membership；
- path membership。

当前默认 `CIPRanker` 仍然是确定性启发式，因此此公式尚未真正控制在线候选排序。

## F04–F05：Agent 动作与 Mask

\[
a=(o,\rho,u)
\]

\[
\pi_\psi^{mask}(a\mid s)
=
\frac{\exp z_\psi(s,a)M_p(s,a)}
{\sum_{a'}\exp z_\psi(s,a')M_p(s,a')}
\]

完整三元动作和 Mask 已实现，但默认主链使用第一版简化策略：

\[
\pi_\psi(o,\rho\mid s)
\]

控制动作 \(u\) 仍主要由规则决定。

## F06–F08：干预和真实改善

\[
Y^{do(h,a)}
\]

通过 released/frozen operations 构造。CP-SAT 对闭包外 mode/start/end 精确固定。

\[
\Delta J^{full}=J(y)-J(y')
\]

Full 验证后按项目词典序目标计算。如果候选没有严格改善，则附加
`NO_TRUE_IMPROVEMENT`，不会更新 incumbent。

当前 `CE` 只包含当前动作的前后差值，没有自动与三种随机对照配对，所以严格因果
责任仍未完成。

## F09–F11：责任价值、成本与多保真

\[
CR_t(h)=\max_a E[\Delta J_t^{full}(h,a)\mathbf1\{Valid=1\}]
\]

当前没有对同一个 CIP 系统执行全部合法动作并估计期望，因此没有严格 CR。

\[
Score(h,a)
=
\frac{E[\Delta J^{full}]P(Valid\mid h,a)}
{Cost(h,a)+\epsilon}
\]

当前 `PosteriorEstimate.acquisition` 使用 gain、validity、uncertainty、risk 与
runtime cost 近似。尚缺 Token 成本与主控制器预算接入。

`mechanisms.py` 对独立干预记录增加了 Beta 有效率、保守收益下界以及
runtime/Token 联合采集值。该后验已有局部测试，但尚未进入默认 controller，也
没有真实项目干预样本，因此不能视为已校准的因果权重。

多保真偏差目前是同类记录的 Light/Full 平均差，不是以 CIP 和动作 embedding 为
输入的 \(b_\ell(h,a)\)。

## F12：完整奖励

\[
\begin{aligned}
r_t=&\alpha\Delta J_t^{full}
+\beta\Delta J_t^{best}
+\gamma E_t^{causal}\\
&-\lambda Cost_t-\rho|\Omega_t|-\mu Risk_t-\xi Invalid_t
\end{aligned}
\]

`full_reward` 和 `light_reward` 已逐项实现。缺口是主控制器尚未构造真实短时程
episode 和 transition，也没有把 Headroom Token 与完整因果对照写入 reward。

## F13–F15：CIP、路径和闭包损失

\[
\mathcal L_{CIP}
=
\lambda_1L_{\Delta J}
+\lambda_2L_{valid}
+\lambda_3L_{rank}
+\lambda_4L_{risk}
+\lambda_5L_{cost}
\]

\[
\mathcal L_{causal}
=
\lambda_PL_{path}
+\lambda_\Omega L_{closure}
+\lambda_SL_{sparse}
\]

当前代码使用 Smooth L1、BCE、多标签 BCE、pairwise logistic loss 和 closure
稀疏均值。函数已经通过前向、反向和一轮训练测试，但尚未使用正式语料训练。

## F16–F18：BC、PPO 与离线 RL

\[
\mathcal L_{BC}=-E\log\pi_\psi(a^*\mid s,h)
\]

BC 已实现并被 CLI smoke 使用。

PPO 包含：

- discounted return；
- advantage；
- clipped ratio；
- value loss；
- entropy bonus；
- gradient clipping。

但默认在线优化没有调用 `policy.update`。

离线 RL 当前只有 advantage-weighted BC loss；IQL/CQL 属于后续对照，不应写成
已经实现。

## F19–F20：条件生成器

目标公式：

\[
\mathcal L_G
=
\mathcal L_{construct}
+\alpha\mathcal L_{repair}
+\beta\mathcal L_{feasible}
\]

当前只实现：

- mode cross entropy；
- normalized start Smooth L1；
- mode legality mask；
- CP-SAT feasible fallback。

这不等于显式实现了完整三项生成器损失。

## F21–F23：安全、单调性和预算

当前满足：

- Full 不通过不接受；
- 目标不严格改善不接受；
- rejected candidate 不覆盖 incumbent；
- best 只在接受时更新；
- Full Oracle 有次数预算。

下一步需要让 posterior acquisition 真正决定哪些 Light 候选升级为 Full。

## F24：时间和 Token 成本

当前：

- 每个 record 保存 runtime；
- posterior 使用平均 runtime；
- 实验指标支持 improvement per second。

待实现：

- Headroom before/after Token；
- input/output/retrieval/cache Token；
- Token 费用；
- Oracle 费用；
- improvement per Token；
- 成本进入 Agent reward 和 Full Oracle acquisition。

## F25：停止条件

当前使用固定 iteration、candidate 和 Full Oracle budget，并在一轮无有效改善时
停止。

目标是：

- 多尺度 VNS 邻域；
- bounded ALNS；
- tabu 去重；
- 所有合法组合覆盖或排除；
- posterior 预期收益阈值；
- 连续平台期；
- 时间、Token、Oracle 预算。

即使完成，也只能称为预算内局部收敛，不能声称全局最优。

## 状态变更要求

更新任意公式状态时，必须同时提供：

1. 对应代码位置；
2. 是否进入默认主闭环；
3. 自动测试或实验 artifact；
4. 当前限制；
5. 下一步条件。

没有 checkpoint 和 dataset hash 时，不得把模型状态改成“已训练”。没有 Full
Oracle 日志时，不得把候选改进写成“已验证”。
