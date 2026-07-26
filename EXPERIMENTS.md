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

## 2026-07-25 — FJSP 分阶段语义链失败诊断与 Opus-5 路由

### 固定案例

- 项目：`wrqccc/FJSP-DRL`
- 固定 commit：`2cf81b13f5044451e78cf780f8fb3e7eeac054c1`
- 预检后输入：16 个有效文件；训练日志、权重和历史结果 Artifact 已排除。

### 失败记录

| 次序 | 阶段 | 现象 | 根因 | 处理 |
|---|---|---|---|---|
| 1 | Opus 配置 | `claude-opus-5` 不受旧账户组支持 | 两组中转配置的通用 `CLAUDE_MODEL` 与旧 Anthropic 凭据被混用 | 模型键按 Provider 分域 |
| 2 | 旧 Opus-4.8 | `Upstream access forbidden` | 旧中转上游不可用 | 不再作为默认链 |
| 3 | 新 Opus-5 HTTP | “仅可用于 CC 官方客户端” | 专用通道拒绝通用 Messages 客户端 | 新增官方 Claude Code CLI Provider |
| 4 | Kimi 首轮 | 多输出 `target_symbol_note` | 严格 schema 服从失败 | 加 schema repair |
| 5 | Kimi 批次 | `project_and_environment_r0` | 擅自修改固定 batch ID | 将精确 ID 纳入 repair |
| 6 | Kimi 修复 | 仍输出 `decisions_oracles_and_unknowns_round_0` | 修复调用仍不服从不可变字段 | Kimi 退出默认主分析 |

Kimi 运行存在轮次和 repair 上限，不是无界死循环；但一次完整调用较慢，且连续
无效修复会造成“长时间运行但无 Artifact”的费用浪费。用户观察到该次尝试费用已
超过 4 元人民币，本实验不把该观察值当作网关精确计费记录。

### 新默认

- Navigator：MiMo-v2.5-pro；
- Analyst / synthesis：Claude Opus-5，经官方 Claude Code CLI；
- 独立影响 Critic：默认 DeepSeek，可显式切换 Opus；
- 每次调用产生实时 `[LLM][START|DONE|ERROR]` 标记和无正文 JSONL 事件日志；
- Opus-5 最小探针成功：约 13.7 秒，客户端报告约 0.028 美元；
- 全套离线测试：48 passed。

## 2026-07-25 — Claude Code 项目 Skill 只读探针

### 迁移来源

本机唯一现有项目 Claude Skill 为 Sortie 的 `sortie3d-harness`。本实验没有复制其
3D 领域内容，只抽取导航、渐进读取、证据审计、有限循环、分层记忆与独立复核方法，
建立 `scheduling-code-semantics`。

Claude 历史日志中只发现一次明确的 `Skill` 工具调用：内置 `/loop`。未发现
`sortie3d-harness` 的显式 Skill 调用记录，因此不把“目录存在”写成“运行时已使用”。

### 两次探针

| 探针 | 调用 | 权限 | 结果 | 成本 |
|---|---|---|---|---|
| 1 | `$scheduling-code-semantics` | 工具可见但未预批准 | Skill 未发现；Read 被 `dontAsk` 拒绝 | `$0.056817` |
| 2 | `/scheduling-code-semantics` | `Read/Glob/Grep` 同时列入 allowedTools | Skill 成功；实际只调用 Read；正确读取项目名/版本 | `$0.06351975` |

第一轮失败原因不是模型不会读代码，而是调用语法和权限配置错误。按 Claude Code
项目 Skill 规则修正后通过。Provider 现已使用 stream-json 记录实际工具名和次数，
不保存工具参数或代码正文。

### 验证

- Skill Creator 校验通过；
- Opus-5 项目 Skill 真实只读探针通过；
- 全套离线测试：49 passed。
# FJSP 项目语义图与继承知识树在线验收（2026-07-26）

## 实验对象

- 仓库：`wrqccc/FJSP-DRL`
- 本地固定提交：`2cf81b13f5044451e78cf780f8fb3e7eeac054c1`
- Navigator：`mimo-v2.5-pro`
- Analyst / Critic：`claude-opus-5`，Claude Code 只读 Skill
- 长期记忆：SQLite v2，车间调度→FJSP→变体继承树

## 结果

| 项目 | 结果 |
|---|---:|
| 问题族 | FJSP |
| 环境判断 | 6 |
| 目标 | 2（其中 1 项明确标记为训练 shaping，而非独立调度目标） |
| 约束 | 6 |
| 决策 | 4 |
| Oracle | 3 |
| 未知/待确认 | 5 |
| 有效证据引用 | 100% |
| 项目图 | 83 节点 / 82 边 |
| 模型调用 | 8 |
| Token | 76,657 |
| Claude Code 只读工具调用 | 26 |
| Schema 修复 | 0 |
| 已报告 Opus 成本 | 约 4.84 USD |

关键判断：

1. 正确区分 FJSP 的固定先验链、候选机器资格、唯一模式选择与机器互斥；
2. 将机器资格和同机排序判为主要优化杠杆；
3. 将 release time=0 判为所有候选共享的退化结构，保留可行性检查但从优化注意中
   降权；
4. 发现代码中的近似 non-delay 动作掩码会收窄可行搜索空间，但它不是 FJSP 的
   问题定义硬约束；
5. 发现归一化加工时间后再以 `op_pt != 0` 判资格，可能在完全柔性实例中与
   OR-Tools 原始资格集合不一致；该项需要人工或定向测试确认。

产物：

- `outputs/fjsp_test/fjsp_opus5_staged_semantics.json`
- `outputs/fjsp_test/fjsp_opus5_review_summary.json`
- `outputs/fjsp_test/fjsp_opus5.events.jsonl`
- `outputs/memory/fjsp_partial_flexibility_resolved.json`

当前结论仍为 `pending_human_review`。100% 证据通过率只表示引用文件和 symbol
存在，不等于所有语义解释已经被人工接受。

后续代码—论文审计确认原结果没有结构化输出次级目标。已从 Critic rationale 重建
6 个候选次级目标和 3 个诊断项，保存在：

- `outputs/fjsp_test/fjsp_opus5_secondary_targets_reconstructed.json`
- `outputs/fjsp_test/FJSP_OPUS5_FINAL_SYNTHESIS_AND_SECONDARY_TARGETS.md`

这些条目仍是 `structured_llm_prior`，不能作为因果结论。

# 同证据 FJSP 大题库召回与 Critic 重放（2026-07-26）

## 实验设计

固定上一节的 `fjsp_opus5_staged_semantics.json`，不重新扫描代码，也不改变问题族、
6 条约束、4 个决策和 evidence packet。只重跑新接入的
`85 项目录 → 确定性召回 → 最多 12 项 → LLM 白名单选择`，以隔离候选库/Harness
变化，避免再次支付原整链 76,657 Token。

## 运行与结果

| 运行 | 结果 | Token / 时间 | 备注 |
|---|---|---:|---|
| 本地 Schema 与回归测试 | 通过 | 0 API Token | 先验证命令、白名单和目录逻辑 |
| MIMO 首次重放 | 通过 | 16,509 / 85.5s | 发现否定句误激活 setup/blocking 变体 |
| MIMO 修正后重放 | 通过 | 15,758 / 87.1s | 15 项合格、12 项入 Prompt、3 项被选择 |
| Opus 路由探针 | 通过 | 36 / 3.9s | 证明模型、Token、Claude Code 客户端和中转路由可用 |
| Opus 调试重放 | 3 次未形成 Artifact | 7,906 / 111.4s；240s；6,138 / 98.5s | 依次暴露 `confidence` 契约、长请求超时和空未知字段 |
| 当前 MIMO final | 通过 | 11,623 / 78.5s | 同一结构化知识和压缩 Prompt；3 个目标、1 个诊断 |
| 当前 Opus final | 通过 | 17,106 / 66.1s | 同条件；5 个目标、4 个诊断；无 Schema 规范化 |

最终 Opus 选择机器选择加工时长机会损失、最大机器工作负荷、关键资源内部空闲时间、
工序间等待总量和稀缺资格负荷；并单列资格一致性、Oracle 参照有效性、Validator 覆盖
和目标下界可信度。程序在完成语义分析后只用结构化 family、constraint、environment
和 objective 激活知识分支，避免“未见 setup”反向打开 setup 分支或注入全部兄弟变体。

产物：

- `outputs/fjsp_test/fjsp_mimo_metric_catalog_replay_v2.json`；
- `outputs/fjsp_test/fjsp_mimo_metric_catalog_replay_v2.events.jsonl`；
- `outputs/fjsp_test/fjsp_opus5_metric_catalog_replay_v2.events.jsonl`（失败 Trace）；
- `outputs/fjsp_test/fjsp_mimo_metric_catalog_replay_final.json`；
- `outputs/fjsp_test/fjsp_opus5_metric_catalog_replay_final.json`；
- `outputs/fjsp_test/FJSP_METRIC_RECALL_COMPARISON_2026-07-26.md`。

结论边界：当前 Opus 是工程约束最完整的高召回先验——它覆盖 MIMO final 的全部三项
并增加等待/关键资源两项，且诊断门更完整。它还不是实验验证的最优集合；在计算器、
候选扰动、Oracle 配对实验和人工真值完成前，不报告 Precision/Recall 或因果结论。

## 高召回候选池重放（2026-07-26）

用户复核后确认本阶段目标是避免漏掉可能相关的次级目标，而不是让 LLM 在真实因素
发现前过早删选。实现因此改为：最多 20 个候选、合理相关项全部保留、三档置信分层、
未知扩展字段隔离留痕，以及程序强制硬约束验证。

| 运行 | 结果 | Token / 时间 | 次级目标 / 诊断 | 备注 |
|---|---|---:|---:|---|
| MIMO high-recall 首次 | Provider 成功、程序安全门拦截 | 12,713 / 97.7s | 未落盘 | 将 hard constraint 标为 not_required；推动安全权回收程序 |
| MIMO high-recall final | 成功 | 12,857 / 83.6s | 8 / 1 | 15 项全部入 Prompt；3 medium、5 low |
| Opus high-recall final | 成功 | 6,590 / 92.4s | 15 / 5 | 3 high、6 medium、6 low；无扩展清理和安全覆盖 |

Opus 的 15 项是待进入计算器、候选变化、干预和 Oracle 校准的候选池。置信度是语义
相关性的未校准软权重，不是因果效应量，也不允许直接控制优化器。

## 题库外开放世界提案重放（2026-07-26）

在相同 FJSP 语义和15项题库候选上增加两条代码证据旁路：新次级目标必须声明候选
排程/构造轨迹范围、候选变化依据和干预把手；新诊断独立承接实现与验证风险。所有
通过项只标记 `proposed`。

| 运行 | Token / 时间 | 目录内目标 | 新次级目标 | 新诊断 | 结果 |
|---|---:|---:|---:|---:|---|
| MIMO open-world 初版 | 13,432 / 95.8s | 7 | 1 | 0 | 发现 non-delay 动作空间压缩率；当时尚未拆诊断通道 |
| Opus open-world 初版 | 8,164 / 118.0s | 15 | 2 | 4 | 暴露资格漂移同时混入 metric/diagnostic 的边界问题 |
| MIMO 双通道 | 14,802 / 105.1s | 5 | 0 | 1 | 成功把资格集合一致性放入诊断通道 |
| Opus 双通道 | 9,050 / 128.0s | 15 | 2 | 3 | 仍有一项兼具诊断性质，推动增加候选变化资格门 |
| MIMO qualified-final | 13,343 / 71.2s | 2 | 1 | 1 | non-delay 掩码活跃比率与资格一致性正确分流 |
| Opus qualified-final | 9,184 / 130.8s | 14 | 2 | 3 | 当前正式开放世界 Artifact |

最终 Opus 的新次级目标是 non-delay 掩码剪除对数量累积（medium）和下界 shaping
增量总量（low）。后者可能因求和望远镜化与终态下界重复，必须先做公式等价审核。
三个新诊断分别覆盖资格集合跨表示一致性、CP-SAT 状态/时限记录和独立可行性校验器。
