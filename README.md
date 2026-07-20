# Schedule Lab

**简体中文** | [English](README_EN.md)

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Version](https://img.shields.io/badge/version-0.1.1-blue)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/tests-25%2F25%20passing-brightgreen)](tests/test_schedule_lab.py)
[![Status](https://img.shields.io/badge/status-active-success)](SCHEDULE_LAB_PLAN.md)
[![Visibility](https://img.shields.io/badge/repository-private-lightgrey)](#license)

面向 JSP、FSP、FJSP、HFSP 和领域约束调度的确定性优化框架。

Schedule Lab 从已有可行调度出发，由 Agent 诊断瓶颈、选择局部优化策略，再由启发式算法、CP-SAT、验证器和领域 Oracle 构造并认证候选。项目重点是可行性、可复现性和可审计性，而不是让大模型直接生成未经验证的调度结果。

## 目录 (Table of Contents)

- [项目简介](#项目简介-overview)
- [核心功能](#核心功能-key-features)
- [当前结果](#当前结果-current-results)
- [系统架构](#系统架构-architecture)
- [安装](#安装-installation)
- [使用方法](#使用方法-usage)
- [MCP 服务](#mcp-服务-mcp-server)
- [项目结构](#项目结构-repository-structure)
- [项目文档](#项目文档-documentation)
- [路线图](#路线图-roadmap)
- [贡献与更新](#贡献与更新-contributing)
- [许可](#许可-license)

## 项目简介 (Overview)

```text
Feasible incumbent
  → normalize and audit
  → diagnose bottlenecks
  → select a bounded neighborhood
  → deterministic repair
  → generic validation
  → optional domain Oracle
  → reproduce and accept or reject
```

设计原则：

- 从可信 incumbent 继续优化，不默认从头随机重排；
- 冻结邻域外决策，并记录 released/frozen 范围；
- Agent 负责诊断和实验编排，不直接宣告可行；
- 求解器生成候选，验证器和 Oracle 负责验收；
- 使用固定 seed、稳定排序、单 worker 和候选哈希保证复现；
- 轨迹联合优化作为可选 Tool，不影响通用调度主流程。

## 核心功能 (Key Features)

| Category | Capabilities | Status |
|---|---|---|
| Problem families | JSP、FSP、FJSP、HFSP、混合领域问题 | Stable |
| Modeling | 多模式工序、可选机器、多资源容量、任意前序、跨资源绑定 | Stable |
| Solvers | Dispatching heuristics、PyJobShop、OR-Tools CP-SAT | Stable |
| Improvement | Compaction、VNS、局部 CP-SAT、因果闭包、有界 ALNS | Stable / v1 |
| Search control | Tabu 哈希、fast → balanced 升级、贝叶斯证据排序 | Stable / v1 |
| Validation | 通用硬约束、冻结区检查、真实指标重算 | Stable |
| Domain Oracle | 舰载机轨迹、避碰、车辆连续性、动态可达性回放 | Integrated |
| Interfaces | Python API、CLI、MCP Server、Codex Skill | Available |
| Visualization | 甘特图、共享时间轴对比视频、审计 manifest | Available |
| Joint trajectories | 路线目录、固定路线时空模型、Oracle Cut | Experimental |
| Agentic RL | 搜索策略控制与图 Encoder | Planned |

## 当前结果 (Current Results)

### 舰载机调度

| 阶段 | 真实 makespan | 验证状态 |
|---|---:|---|
| Greedy 基线 | 675.5 s | 已保存、可复现 |
| 受控策略搜索 | 637.5 s | 通用验证与领域验证通过 |
| 确定性 ALNS 第 1 轮 | 636.2 s | 已验证 |
| 确定性 ALNS 第 2 轮 | 630.5 s | 已验证 |
| **当前 incumbent** | **627.8 s** | **已验证** |

当前 incumbent 相比 675.5 秒基线缩短 **47.7 秒 / 7.06%**。

![Current 627.8-second carrier schedule](outputs/carrier_alns_best_iter3_gap6_closed_630_5.png)

关键产物：

- [当前 627.8 秒调度方案](outputs/carrier_alns_best_iter3_gap6_closed_630_5.json)
- [原始 675.5 秒基线](outputs/carrier_greedy_baseline_675_5.json)
- [637.5 vs 627.8 对比视频](outputs/videos/carrier_schedule_comparison_637_5_vs_627_8.mp4)
- [对比视频审计 manifest](outputs/videos/carrier_schedule_comparison_637_5_vs_627_8.manifest.json)

> 文件名中的 `630_5` 为历史审计标记，表示该轮优化的输入 incumbent 是 630.5 秒。文件内调度按最晚工序结束时间重新计算后的真实 makespan 为 627.8 秒。

### 多问题族回归

| 问题族 | LPT incumbent | 改进结果 | 结论 |
|---|---:|---:|---|
| JSP | 14 | **11** | 已改善 |
| FSP | 22 | **19** | 已改善 |
| FJSP | 11 | **7** | 已改善 |
| HFSP | 17 | 17 | 当前有界邻域未改善 |

完整结果见[自适应多问题族回归](outputs/adaptive_multifamily_improvement_benchmark.json)。

### 被拒绝的实验候选

| 实验 | 抽象结果 | 领域结果 | 结论 |
|---|---:|---:|---|
| 固定路线时空 CP-SAT | 619.5 s | 803.9 s，出现 40 个绑定变化 | 拒绝 |
| 有界路线绑定主问题 | 634.6 s | 未送入昂贵领域 Oracle | 在 Oracle 前拒绝 |

抽象求解器得到的目标值，只有通过全部通用验证和领域回放后，才能作为正式优化结果。

## 系统架构 (Architecture)

```text
Problem Adapter
└── Canonical Scheduling IR
    ├── Metrics and Bottleneck Diagnosis
    ├── Family Strategy Router
    │   ├── JSP: critical blocks
    │   ├── FSP: permutation and blocking
    │   ├── FJSP: routing and sequencing
    │   └── HFSP: stage load and sink gaps
    └── Deterministic Search Controller
        ├── VNS and exact enumeration
        ├── CP-SAT local repair
        ├── bounded ALNS
        └── tabu and Bayesian budget allocation
            ↓
      Generic Validator
            ↓
      Optional Domain Oracle
            ↓
      Reproduce → Compare → Accept / Reject
```

### 职责边界

| 组件 | 职责 |
|---|---|
| Agent | 诊断瓶颈，选择策略、邻域和预算 |
| 启发式 / VNS / ALNS | 生成结构化候选提案 |
| CP-SAT / 精确方法 | 在释放区域内构造合法调度 |
| 通用验证器 | 检查前序、资源、资格、绑定和冻结决策 |
| 领域 Oracle | 检查轨迹、碰撞、状态依赖可达性和真实时间 |
| 人工审核 | 确认目标、风险阈值和正式发布决策 |

## 安装 (Installation)

### 环境要求

- Python 3.11 or later
- macOS, Linux or Windows
- 本私有仓库的访问权限

### 安装步骤

```bash
git clone https://github.com/wgy577/schedule-lab.git
cd schedule-lab

python3 -m venv .venv
.venv/bin/pip install .
```

舰载机专用命令还需要本地 legacy 项目、训练网络权重和 MAT 轨迹资源。通用 JSP/FSP/FJSP/HFSP 工作流不依赖这些资产。

## 使用方法 (Usage)

### 运行测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

### 运行调度回归

```bash
.venv/bin/schedule-lab benchmark
```

### 运行自适应多问题族回归

```bash
.venv/bin/schedule-lab adaptive-improvement-benchmark --baseline-rule lpt
```

### 分析并改进 incumbent

```bash
.venv/bin/schedule-lab improvement-workflow \
  problem.json incumbent.json \
  --evidence-count 12 \
  --validated-elite-count 2 \
  --output outputs/improvement_workflow.json
```

### 审计舰载机调度

```bash
.venv/bin/schedule-lab carrier-audit
```

该审计命令为只读操作，不会覆盖当前方案。

### 生成对比视频

```bash
python3 workflows/video/render_schedule_comparison.py
```

视频工作流使用统一时间尺度。较短方案完成后停留在最终帧并等待较长方案结束，两侧不会被分别压缩到相同时长。

## MCP 服务 (MCP Server)

启动服务：

```bash
.venv/bin/schedule-lab-mcp
```

主要工具包括：

- `scheduling_capabilities`
- `analyze_schedule`
- `solve_problem`
- `compare_schedules`
- `plan_schedule_improvement`
- `fast_improve_schedule`
- `adaptive_improve_schedule`
- `plan_joint_schedule_and_trajectories`
- `audit_current_carrier`
- `search_current_carrier`

MCP 层用于向 Agent 提供调度能力，不替代求解器、验证器或领域 Oracle。

## 项目结构 (Repository Structure)

```text
schedule-lab/
├── README.md
├── README_EN.md
├── CHANGELOG.md
├── EXPERIMENTS.md
├── SCHEDULE_LAB_PLAN.md
├── pyproject.toml
├── run_schedule_lab.py
├── src/schedule_lab/
│   ├── model.py
│   ├── validation.py
│   ├── fast_controller.py
│   ├── generic_neighborhood.py
│   ├── strategy_router.py
│   ├── carrier_*.py
│   └── solvers/
├── tests/
├── skills/
├── workflows/
└── outputs/
```

## 项目文档 (Documentation)

| 文档 | 用途 |
|---|---|
| [技术计划与路线图](SCHEDULE_LAB_PLAN.md) | 当前能力、架构、风险和分阶段开发计划 |
| [实验记录](EXPERIMENTS.md) | 已接受、已拒绝、临时和计划中的实验 |
| [更新日志](CHANGELOG.md) | 版本化仓库更新 |
| [调度优化 Skill](skills/improve-schedules-with-oracles/SKILL.md) | 可复用 Agent 工作流和验证规则 |
| [方法选择](skills/improve-schedules-with-oracles/references/method-selection.md) | 问题族与优化方法路由 |
| [高级方法组合](skills/improve-schedules-with-oracles/references/advanced-optimization-portfolio.md) | 分解、路径重连、Oracle Cut 和鲁棒优化 |
| [调度—轨迹联合设计](skills/improve-schedules-with-oracles/references/agentic-rl-and-joint-trajectories.md) | 可选轨迹 Tool 和 Agentic RL 边界 |
| [对比视频规范](workflows/video/COMPARISON_VIDEO_TEMPLATE.md) | 共享时间轴渲染和审计要求 |

## 路线图 (Roadmap)

- [x] Unified JSP/FSP/FJSP/HFSP problem representation
- [x] Deterministic validation and metric audit
- [x] VNS, local CP-SAT and bounded ALNS workflow
- [x] Carrier trajectory/collision Oracle integration
- [x] Tabu signatures, Oracle Cuts and Bayesian evidence ranking
- [x] Validated carrier improvement from 675.5 to 627.8 seconds
- [ ] Standardize the 627.8-second incumbent artifact and reproduction manifest
- [ ] Compare causal closure, local branching and shifting bottleneck on the current incumbent
- [ ] Expand the multi-family benchmark suite
- [ ] Package trajectory optimization as a fully optional Tool
- [ ] Add robustness scenarios and lexicographic multi-objective acceptance
- [ ] Evaluate Agentic RL and graph encoders after sufficient validated evidence exists

详细里程碑和验收标准见 [SCHEDULE_LAB_PLAN.md](SCHEDULE_LAB_PLAN.md#8-后续技术计划)。

## 贡献与更新 (Contributing)

本仓库以“已验证实验”为更新单位，不以未经验证的求解器输出作为正式结果。

提交改动前：

1. 保留当前 incumbent 并记录规范化哈希；
2. 在 [EXPERIMENTS.md](EXPERIMENTS.md) 登记实验；
3. 保持声明邻域以外的所有决策冻结；
4. 运行通用验证和所需领域 Oracle；
5. 对接受候选进行复跑并比较规范化哈希；
6. 更新 [CHANGELOG.md](CHANGELOG.md) 的 `Unreleased` 部分；
7. 运行完整测试集。

实验状态：

- `ACCEPTED`：全部验证通过且目标严格改善；
- `REJECTED`：不可行、恶化、不稳定或无法复现；
- `PROVISIONAL`：仅在抽象模型中成立，仍等待领域验证；
- `PLANNED`：已定义但尚未执行。

## 许可 (License)

本仓库为私有研究项目，目前未授予公开使用许可。未经仓库所有者允许，不得重新分发源代码、模型、轨迹资产或实验产物。
