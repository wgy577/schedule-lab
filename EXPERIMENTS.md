# 实验记录

## 2026-07-24 — 独立通用研究平台 smoke

### 验收目的

本轮只验证工程闭环、约束安全和可复现性，不将小实例结果包装成论文性能结论。
项目运行时没有导入任何外部业务调度仓库。

共同设置：

- 四类问题共享同一 IR、图、CIP、Agent、repair 和 Oracle；
- 固定 seed，CP-SAT 单 worker；
- 从 incumbent 出发，闭包外 mode/start/end 精确冻结；
- 每个实例 1 次外层迭代，最多评估 2 个候选；
- Code→Static→Light→Full 逐级验证；
- 只有词典序目标严格改善才更新 incumbent；
- rejected candidate 同样记录。

### 自动验证

```text
10 passed
```

覆盖：

- JSP/FSP/FJSP/HFSP 确定性 baseline；
- bounded CP-SAT 与闭包外冻结；
- time-window 与 no-wait 编码；
- 语义编译证据门；
- GNN 多任务头与反向传播；
- 条件部分排程编码；
- 层级动作 mask；
- 统计检验与阶段依赖；
- 完整控制器 smoke。

### 四问题族结果

| 问题族 | 初始 makespan | smoke 最好值 | 正式含义 |
|---|---:|---:|---|
| JSP | 14 | 11 | 找到并通过完整通用验证 |
| FSP | 27 | 27 | 无严格改善，保留 incumbent |
| FJSP | 14 | 8 | 找到并通过完整通用验证 |
| HFSP | 23 | 23 | 无严格改善，保留 incumbent |

结果：

- `outputs/jsp_optimization.json`
- `outputs/fsp_optimization.json`
- `outputs/fjsp_optimization.json`
- `outputs/hfsp_optimization.json`
- `outputs/*_experiments.jsonl`
- `outputs/semantic_compilation.json`

### 尚未声明

- 尚未在 OR-Library、Taillard、Brandimarte 等完整公开基准上训练和评估；
- 尚未训练可发布的跨实例 GNN/PPO checkpoint；
- 尚未执行全部消融、leave-one-family-out 与统计显著性报告；
- 当前数字只证明代码链条能发现、验证、接受或拒绝候选，不证明全局最优。

## 2026-07-24 — 火山 Coding Plan/GLM-5.2 语义编译验收

### 目的

验证真实 LLM Provider、选项库、严格 JSON、Schema 校验和证据引用链。输入为
当前项目的 10,000 字符脱敏证据包，`.env` 未进入请求。

### 结果

| 项目 | 结果 |
|---|---|
| 请求模型 | `ark-code-latest` |
| 服务端实际模型 | `glm-5.2` |
| 输入/输出/总 Token | 8,662 / 6,803 / 15,465 |
| 延迟 | 80.288 秒 |
| Provider 尝试 | 1 |
| Schema 修复 | 0 |
| 识别目标/约束/决策/Oracle | 6 / 8 / 4 / 4 |
| 证据引用检查 | 51/51 通过 |
| 输出状态 | `pending_human_review` |

Artifact：

`outputs/llm_semantic_compilation.json`

### 结论边界

- 证明火山 Coding Plan、GLM-5.2、答题卡 Schema 和证据审计可以连通；
- 证据检查只证明引用路径和 symbol 存在，不证明业务解释绝对正确；
- 当前是有限证据包上的单轮分析，不代表大型项目多轮语义编译已完成；
- 输出没有自动覆盖正式 `ProjectSemantics`，必须经人工审核。

## 2026-07-24 — 网上 L2D 论文—代码语义盲测

### 案例与隔离

- 案例：NeurIPS 2020 L2D JSSP；
- 论文来自 NeurIPS 官方 PDF；
- 代码来自作者公开的 L2D 仓库；
- 仓库固定 commit：
  `7b2efbb1ffc960260b16952f2bed68e500765bf0`；
- 论文只用于 draft 隐藏标签；
- 代码读取规划和代码分析请求不含论文文本或标签；
- 仓库 `paper/` 在 inventory 与证据包中排除。

### API 与成本

| 阶段 | 实际模型 | Token | 延迟 |
|---|---|---:|---:|
| 论文 draft 标签 | GLM-5.2 | 22,650 | 116.596 秒 |
| 代码读取规划 | GLM-5.2 | 2,477 | 17.868 秒 |
| 代码语义识别 | GLM-5.2 | 22,094 | 191.610 秒 |
| 合计 | — | 47,221 | 326.073 秒 |

代码识别阶段发生 2 次 Provider 尝试，没有 Schema 修复。

### 分类结果

| 语义组 | F1 | Exact |
|---|---:|---|
| 项目类型 | 0.000 | 否 |
| 问题族 | 1.000 | 是 |
| 环境 | 1.000 | 是 |
| 目标 | 1.000 | 是 |
| 约束 | 0.750 | 否 |
| 决策 | 0.500 | 否 |
| Oracle | 0.000 | 否 |
| 允许干预 | 0.667 | 否 |
| **Macro / exact rate** | **0.615** | **0.375** |

论文引用全部通过；代码证据 30/30 通过。该证据结果只证明引用文件与 symbol
存在。论文标签状态仍为 `draft_llm_label`，因此这个分数是 Harness 开发基线，
不是对模型能力的最终论文结论。

Artifact：`outputs/l2d_semantic_harness.json`。

## 2026-07-24 — L2D 五模型代码语义对比

使用同一份 L2D draft 隐藏标签、同一 commit、42,000 字符代码预算，比较
GLM-5.2、DeepSeek-v4-pro、MiMo-v2.5、Kimi-k3 与 Claude-Opus-4.8。

| 模型 | Macro-F1 | Exact | 证据 | Token | 延迟 |
|---|---:|---:|---:|---:|---:|
| Kimi-k3 | 0.653 | 0.250 | 1.000 | 41,011 | 453.05 秒 |
| Claude-Opus-4.8 | 0.628 | 0.375 | 1.000 | 15,225 | 80.36 秒 |
| GLM-5.2 | 0.615 | 0.375 | 1.000 | 24,571 | 209.48 秒 |
| MiMo-v2.5 | 0.576 | 0.375 | 0.529 | 20,124 | 27.07 秒 |
| DeepSeek-v4-pro | 0.478 | 0.250 | 0.909 | 25,128 | 80.97 秒 |

Kimi 第一次正式运行因网络读取超时失败，第二次成功时发生一次 Schema 修复；
其接口强制 `temperature=1`。MiMo 虽最快，但有 8 条引用无效。Opus 一次完成、
证据全部通过且 Token 最少，是当前单案例下最均衡的默认主模型。

完整逐字段结果、限制和推荐路由见
`docs/semantics/L2D_MODEL_COMPARISON_REPORT.md`。标签仍为 draft，不能把本表解释为正式
模型排行榜。

## 2026-07-24 — 最高质量推理协议与 L2D 人工真值门

### Provider 协议

语义编译请求统一改为质量优先：

| Provider | 当前模型 | 推理设置 | 在线 smoke |
|---|---|---|---|
| 火山 | 网关请求 `ark-code-latest`，返回 GLM-5.2 | thinking enabled，effort max | 通过 |
| DeepSeek | `deepseek-v4-pro` | thinking enabled，effort max | 通过 |
| Kimi | `kimi-k3` | 始终推理，effort max | 通过 |
| MiMo | `mimo-v2.5-pro` | thinking enabled，接口最高 effort high | 通过 |
| Claude | `claude-opus-4-8` | adaptive thinking，effort max | 配置完成；中转上游暂时不可用 |

MiMo 的协议只接受 `low/medium/high`，因此统一请求 `max` 时显式映射为该接口最高
合法档 `high`。历史对比表中的 MiMo-v2.5 结果没有被改写成 Pro 结果。

### 人工真值门

复核发现旧 L2D draft 标签不能直接承担 ground truth：

- 官方论文正文与仓库内“正文+补充材料”PDF 页数和哈希不同；
- paper claim、code fact、实验 baseline 和性能主张需要分层；
- release time、OR-Tools/PDR 基线、直接与派生决策存在待审核差异；
- JSP/JSSP、capacity-1/no-overlap 等语义等价项需要规范化。

已建立 `docs/semantics/L2D_PAPER_CODE_HUMAN_VERIFICATION.md`。人工审核完成前不重跑正式
模型排名，不根据旧 F1 调 Prompt，也不把 draft 标签用于训练。

## 2026-07-25 — 调度问题族知识检索与约束影响 Critic

### 用户确认的口径

- L2D 按最普通的静态 JSSP 处理；
- release time 在该实现中固定为 0，不进入当前优化注意力；
- OR-Tools、传统 PDR、benchmark 和硬件属于实验比较，不属于核心调度约束；
- 选择工序后最早可行插入属于经典构造规则。

### 实现

- 新增 JSP/FSP/HFSP/FJSP、常见变体及八类工程约束模式知识种子；
- 使用精确 family hint、短缩写边界和变体词法命中；
- 检索结果默认进入 LLM 语义编译，但禁止充当项目 evidence；
- 新增独立高推理 Constraint Critic；
- 将影响拆为 feasibility、decision leverage、objective sensitivity 和
  candidate discrimination；LLM 只选择枚举档位，数值由程序固定映射；
- 程序禁止把任何 `hard=true` 约束移出验证。

### 验证

离线测试覆盖 JSP/FJSP 消歧、FJSP transport 变体和 AGV 工程模式命中、知识先验注入、项目证据
隔离，以及“release time 优化杠杆为 0、但 must_validate 仍为 true”的安全门。

当前未声明：

- 尚无成功的在线 Constraint Critic Artifact；
- 尚未用人工标签校准四维权重；
- 权重尚未进入 CIP、邻域或 Agent 控制器；
- 尚未证明减少 Token 或提高跨项目识别率。
