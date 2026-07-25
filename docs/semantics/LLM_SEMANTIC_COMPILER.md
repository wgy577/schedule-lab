# LLM 项目语义编译器

> 当前阶段：单轮主链已真实验收；分阶段 Agent 链代码已实现并完成离线验收  
> 导航 Provider：默认 `MIMO_*`，也可替换为 DeepSeek 等低成本模型  
> 强分析 Provider：默认 Anthropic/Opus 协议，也可切换 `SEED_*` 等 OpenAI 兼容模型  
> 安全原则：LLM 负责理解，程序负责索引与核验，人工负责最终业务确认

## 0. 新的分阶段语义链

`compile-semantics-staged` 不再让一个模型拿到静态证据包后一次作答。当前实现链为：

```text
仓库文件枚举
  → 排除论文、密钥、二进制和构建产物
  → Python AST 提取符号、import、call 和代码角色先验
  → 低成本 Navigator 按 70k 字符预算分片建图
  → 分片导航合并为统一 ProjectNavigation
  → 强模型处理批次 1：项目类型 / 问题族 / 环境 / 数据
  → 强模型处理批次 2：目标 / 硬约束 / 实现路径
  → 强模型处理批次 3：决策 / Oracle / 冲突 / 未知
  → 每批先回答“证据是否充分”选择题
  → 程序批准或拒绝相关函数复读请求
  → 批次间只传递结构化压缩短期记忆
  → 最终综合模型重新引用完整证据账本
  → JSON Schema 修复
  → 文件、路径和 symbol 证据审计
  → 可选独立 Constraint Impact Critic
  → pending_human_review
```

这条链实现在 `src/causal_schedule_lab/semantic_agent.py`。旧的
`compile-semantics-llm` 保留，用于回归比较，不会被静默替换。

### 0.1 Prompt 在什么时候注入

| 阶段 | 模型 | System Prompt 只规定什么 | User Prompt 注入什么 | 输出 |
|---|---|---|---|---|
| 导航分片 | 低成本模型 | 只画地图、禁止业务结论、禁止发明路径 | 本分片文件角色、符号、import、带行号摘录 | 分片模块图 |
| 导航合并 | 同一低成本模型 | 同上 | 完整文件目录 + 所有分片图 | `ProjectNavigation` |
| 语义批次 | 高能力模型 | 本批问题、证据充分性选项、读请求规则 | 导航 + 当前批文件 + 短期记忆 + 长期记忆先验 + 新工具证据 | `BatchAnalysis` |
| 工具复读后续轮 | 同一高能力模型 | 不变 | 上轮压缩状态 + 本轮新证据，不重放全部历史推理 | 更新后的批次答卷 |
| 最终综合 | 高能力模型 | 完整语义选项和证据规则 | 最终证据账本 + 压缩批次记忆 + 导航 | `SemanticAnalysis` |
| 影响 Critic | 可独立高能力模型 | 影响权重与硬约束安全门 | 已审计语义 + 问题族知识 | 影响报告 |

### 0.2 三种上下文不是同一种“记忆”

- `ProjectNavigation`：仓库地图，只用于定位；
- `ShortTermMemory`：前三批已经确认的事实、证据 ID、矛盾和未知；不保存自由推理；
- `LongTermMemoryContext`：SQLite 属性图和 L0–L4 FTS5 检索结果，只是经典问题族和
  工程模式先验，不得被引用为当前项目证据；
- `ToolEvidence`：程序批准后从当前仓库重新读取的精确函数或文件片段，进入最终证据账本。

只有最后一种和初始当前项目文件证据可以支撑 `SemanticAnalysis.evidence`。

### 0.3 复读工具的决定权

模型只能提交结构化 `EvidenceReadRequest`，不能执行 shell、任意路径读取或自行扩大
上下文。每个问题必须先选择：

```text
sufficient | partial | insufficient | contradictory | not_applicable
```

程序按固定顺序审批：

1. 请求必须被某个证据评估引用；
2. `sufficient` 和 `not_applicable` 不允许继续读；
3. 文件必须在项目白名单内，且不是论文、秘密或二进制；
4. 同一文件/符号不能重复读取；
5. symbol 必须真实存在；
6. 源与目标之间必须存在同文件、AST call、import、配置或 Validator/Oracle 关系；
7. `low` 优先级默认延后；
8. 每轮最多 4 次、每次编译默认最多 18 次；
9. 每次批准、拒绝及理由都写入 Artifact。

批准后，工具按 AST 的 `lineno/end_lineno` 读取目标函数，并最多补充三个同文件被调用
函数。它不是一个开放式“让模型随便翻仓库”的工具。

## 1. 输入

### 1.1 当前内部 smoke 输入

程序先建立脱敏代码证据包：

- `README`、项目清单和构建配置；
- JSON/YAML/TOML 配置；
- 源代码顶层类型、函数和关键片段；
- 测试代码；
- 项目文档；
- 文件相对路径、SHA-256 和带行号摘录。

默认排除：

- `.env` 和 `.env.*`；
- `.git`、虚拟环境和依赖目录；
- 构建产物；
- outputs 和 checkpoints；
- 二进制文件。

旧单轮链使用字符预算选择证据。新链已经实现分片导航、分批归纳和最多三轮受控复读。
正式多模型质量评测尚未执行，因此当前状态仍是“代码已实现但未完成真实在线验收”。

首次真实运行分析的是本项目 `causal_schedule_lab` 自身，不是外部论文项目。由于
10,000 字符预算，当次只包含：

- `pyproject.toml`；
- `README.md`；
- `configs/base_semantics.json`；
- `configs/generic_semantics.json`；
- `configs/training.json`。

因此该次运行只证明接口和结构化输出链路，不代表完整仓库理解。

### 1.2 正式调度案例的标准输入

以后优先选择同时具备论文与代码的调度项目。一个案例包建议包含：

```text
case/
├── papers/                 # 论文 PDF、补充材料、附录
├── repository/             # 对应代码仓库固定 commit
├── instances/              # 训练、验证、测试实例
├── results/                # 论文表格或作者发布结果
├── checkpoints/            # 可选
└── CASE_MANIFEST.json      # 来源、版本、许可证和对应关系
```

论文主要提供：

- 问题定义；
- 环境与状态假设；
- 目标函数；
- 硬约束；
- 决策变量；
- 算法流程；
- 数据划分；
- 评价指标和基线；
- 实验硬件与预算；
- 消融设计。

代码主要提供：

- 论文公式的实际落点；
- 环境初始化和状态转移；
- 约束检查器；
- reward/objective 实现；
- 默认参数；
- 数据预处理；
- 求解器和模型调用；
- 终止条件；
- seed 和并行策略；
- 论文未说明的工程假设。

### 1.3 Paper→Code 对齐矩阵

LLM 先读论文形成 `PaperSemantics`，再读取代码形成 `CodeSemantics`，最后逐项
对齐。每项只能选择：

| 状态 | 含义 |
|---|---|
| `paper_and_code` | 论文声明且代码有对应证据 |
| `paper_only` | 论文声明，但代码暂未找到对应实现 |
| `code_only` | 代码存在，但论文没有说明 |
| `conflict` | 论文和代码在公式、参数、约束或流程上冲突 |
| `unknown` | 当前材料不足 |

正式 `ProjectSemantics` 优先使用 `paper_and_code`。`paper_only`、`code_only` 和
`conflict` 必须进入人工审核，不能静默合并。

论文是训练和标注阶段的特权信息，不是生产部署依赖。无论文与碎片化文档场景的
Teacher–Student、模态缺失训练和评估方案见
[CODE_ONLY_SEMANTIC_LEARNING.md](CODE_ONLY_SEMANTIC_LEARNING.md)。

## 2. LLM 需要解析什么

在解析前，系统先从
`src/causal_schedule_lab/knowledge/scheduling_families.json` 检索 JSP、FSP、
HFSP、FJSP 的经典定义和常见变体。知识条目只作为先验，不能替代项目 evidence。
模型应重点识别项目相对经典问题的增量，而不是在每个案例上重新推理所有基础约束。

### 2.1 选择题字段

| 字段 | 主要选项 |
|---|---|
| 项目类型 | scheduling、optimization、simulation、ML、data、web、scientific、mixed、other、unknown |
| 调度问题族 | JSP、FSP、FJSP、HFSP、RCPSP、JSSP、VRP、other、unknown |
| 环境 | static/dynamic、deterministic/stochastic、offline/real-time、simulation-backed 等 |
| 目标 | makespan、tardiness、flow time、cost、energy、throughput、robustness 等 |
| 方向 | minimize、maximize、satisfy、unknown |
| 约束 | precedence、capacity、eligibility、time window、transport、collision 等 |
| 约束范围 | global、project、job、operation、resource、route、time 等 |
| 决策 | select、assign resource、sequence、start time、release、route、batch 等 |
| Oracle | validator、solver、simulator、digital twin、service、human 等 |
| 置信度 | high、medium、low |

模型不得创造枚举值。无法归类时只能选择 `other` 或 `unknown`。

### 2.2 允许自由文本的字段

自由文本只用于：

- 项目摘要；
- 目标、约束和决策的具体含义；
- 证据说明；
- 无法确认的问题；
- 建议补充的证据。

自由文本不能代替分类选项，也不能直接成为已验证事实。

## 3. 输出 JSON

核心结构：

```json
{
  "schema_version": "1.0",
  "language": "zh-CN",
  "summary": "...",
  "project_type": "scheduling_optimization",
  "problem_families": ["JSP"],
  "environments": [],
  "objectives": [],
  "constraints": [],
  "decisions": [],
  "oracles": [],
  "allowed_interventions": [],
  "unknowns": [],
  "overall_confidence": "medium"
}
```

每个目标、约束、决策和环境判断必须带：

```json
{
  "evidence": [
    {
      "file": "src/example.py",
      "symbol": "validate_schedule",
      "detail": "该函数检查资源重叠"
    }
  ]
}
```

## 4. 稳定性机制

新增知识检索控制：

- 短问题族术语使用边界匹配，避免 JSP 误命中 FJSP；
- 初始词法检索后，根据 LLM 已确认 family 再做精确检索；
- 经典知识与项目 evidence 分离，Prompt 禁止引用知识库作为项目事实；
- 对比算法、benchmark、硬件和性能主张不进入调度硬约束。

1. 温度固定为 `0`；
2. Prompt 给出完整选项库；
3. Pydantic 生成并校验 JSON Schema；
4. 禁止额外字段；
5. ID 使用固定模式；
6. 证据字段必填；
7. JSON 无法解析时，从文本中提取第一个合法对象；
8. Schema 不通过时允许一次定向修复；
9. 修复仍失败则整个任务失败，不放宽 Schema；
10. 所有 LLM 结果保持 `pending_human_review`。

分阶段链额外提供：

11. 低成本导航与高能力分析使用两个独立 Provider；
12. 导航输入按 70,000 字符近似预算分片，避免单次吞下整个仓库；
13. 三个语义批次固定，不让模型自行改变任务边界；
14. 每批先做证据充分性 MCQ，再允许申请复读；
15. 复读由 AST 关系和确定性预算门审批；
16. 后续轮只收到压缩短期记忆和新增证据；
17. 最终综合必须重新引用证据账本，不能引用导航或长期记忆；
18. Navigator、Analyst、读请求和 Token 分开记账。

## 5. 证据审计

程序检查：

- 引用文件是否在本次证据包；
- 文件是否仍存在于项目目录；
- 路径是否越出项目根目录；
- 引用 symbol 是否存在于索引。

“证据引用有效”只说明引用存在，不等于业务声明真实。隐含约束、未文档规则和
业务优先级必须由人审核。

## 6. Provider 配置

`.env`：

```dotenv
SEED_API=
SEED_BASE_URL_OPENAI=https://ark.cn-beijing.volces.com/api/coding/v3
SEED_MODEL=ark-code-latest
```

密钥不会进入证据包、JSON Artifact 或日志。

除 OpenAI 兼容接口外，当前还提供 Anthropic Messages 兼容 Provider，读取：

```dotenv
ANTHROPIC_AUTH_TOKEN=
ANTHROPIC_BASE_URL=
CLAUDE_MODEL=
```

已对 GLM、DeepSeek、MiMo、Kimi 和 Claude 中转接口完成真实调用。语义编译主线
现在使用质量优先协议，不再为了延迟关闭推理：

| 模型 | 当前模型 ID | 最高质量协议 |
|---|---|---|
| 火山 Coding Plan | 网关请求 `ark-code-latest`，当前返回 GLM-5.2 | `thinking.enabled` + `reasoning_effort=max` |
| DeepSeek | `deepseek-v4-pro` | `thinking.enabled` + `reasoning_effort=max` |
| Kimi | `kimi-k3` | 模型始终推理，不发送 `thinking`；`reasoning_effort=max` |
| MiMo | `mimo-v2.5-pro` | `thinking.enabled` + 接口支持的最高档 `reasoning_effort=high` |
| Claude | `claude-opus-4-8` | adaptive thinking + `output_config.effort=max` |

对于推理模型，Provider 不再发送可能覆盖其推荐推理配置的 temperature。请求会同时
记录 requested/effective reasoning effort、thinking mode、可见的 reasoning 字符数
和网关返回的 thinking token。MiMo 接口只接受 `low/medium/high`，所以统一请求的
`max` 会被明确映射为该接口的最高合法值 `high`，不是静默降级。

DeepSeek、Kimi、MiMo 和火山网关已完成该协议的在线 smoke；Claude 中转已按
Opus-4.8 的 adaptive/max 协议配置，但本轮在线确认遇到中转上游暂时不可用，需在
服务恢复后补一次 smoke。中转接口返回的模型名属于外部信任边界，系统无法独立
验证其背后的真实权重。

## 7. 命令

先检查证据包，不调用 API：

```bash
PYTHONPATH=src .venv/bin/python -m causal_schedule_lab.cli \
  compile-semantics-llm \
  --project-root . \
  --dry-run \
  --output outputs/llm_semantic_dry_run.json
```

调用火山 Coding Plan 的旧单轮回归链：

```bash
PYTHONPATH=src .venv/bin/python -m causal_schedule_lab.cli \
  compile-semantics-llm \
  --project-root . \
  --env-file .env \
  --provider-prefix SEED \
  --output outputs/llm_semantic_compilation.json
```

分阶段链先做不调用 API 的仓库清单检查：

```bash
PYTHONPATH=src .venv/bin/python -m causal_schedule_lab.cli \
  compile-semantics-staged \
  --project-root /path/to/project \
  --dry-run \
  --output outputs/staged_semantic_dry_run.json
```

默认用 `MIMO_*` 导航、Anthropic/Opus 做强分析：

```bash
PYTHONPATH=src .venv/bin/python -m causal_schedule_lab.cli \
  compile-semantics-staged \
  --project-root /path/to/project \
  --navigator-provider-prefix MIMO \
  --analyst-protocol anthropic \
  --max-rounds-per-batch 3 \
  --max-reads 18 \
  --output outputs/staged_semantic_compilation.json
```

若强分析也走 OpenAI 兼容网关：

```bash
PYTHONPATH=src .venv/bin/python -m causal_schedule_lab.cli \
  compile-semantics-staged \
  --project-root /path/to/project \
  --navigator-provider-prefix DEEPSEEK \
  --analyst-protocol openai \
  --analyst-provider-prefix SEED
```

当前 `.env` 的具体密钥、模型 ID 和中转身份仍是外部信任边界，不写入 Artifact。

## 8. 当前限制与下一步

新增的约束影响 Critic 分别输出可行性重要度、决策杠杆、目标敏感度和候选区分度。
四个维度均从有限档位选择，程序再固定映射为排序分数；LLM 不填写自由小数。
硬约束即使优化杠杆为 0 也不能移出验证。该 Critic 已有可选 CLI 和离线测试，但
尚未完成多案例在线校准，当前不能自动影响 CIP 或搜索邻域。详细设计见
[SCHEDULING_SEMANTIC_KNOWLEDGE_AND_IMPACT.md](SCHEDULING_SEMANTIC_KNOWLEDGE_AND_IMPACT.md)。

当前已经实现：

- OpenAI 兼容 Provider；
- `.env` 安全读取；
- HTTPS 和超时限制；
- 重试与退避；
- JSON mode 兼容回退；
- 选择题 Schema；
- 证据包；
- JSON 解析与一次修复；
- 证据路径/symbol 审计；
- Token、模型、延迟和 request ID 元数据。
- 大仓库分片 Navigator 与统一导航合并；
- 三个固定语义分析批次；
- 证据充分性 MCQ 与受控 AST 函数复读；
- 压缩短期记忆和可选图谱/FTS 长期记忆先验；
- 复读批准/拒绝审计和最终证据账本综合。

尚未实现：

- 分片模块的递归层级 Map/Reduce（当前只有一层分片 + 一次合并）；
- 批次间自动矛盾裁决（当前会记录 contradiction，不会替模型选择一方）；
- 人工 approve/reject/annotate CLI；
- 将审核结果转换成正式 `ProjectSemantics`；
- Provider 级总费用预算和熔断状态持久化；
- 各 API 阶段的即时 checkpoint 与失败后断点恢复；
- 新旧 Prompt 链与多模型的系统化在线评测集。

已经在独立 Harness 中实现：

- 论文 PDF 文本提取与 draft `PaperSemanticLabel`；
- 代码 inventory 与 LLM 文件读取规划；
- 论文目录双重排除和代码白名单证据包；
- Paper label 与 Code prediction 的八类集合评分；
- 各阶段模型、Token、调用次数和延迟记录。

详见 [LLM_SEMANTIC_HARNESS.md](LLM_SEMANTIC_HARNESS.md)。这不是最终
`PaperSemantics ↔ CodeSemantics` 冲突合并器，也没有跳过人工审核。

## 9. 首次真实运行基线

`2026-07-24` 使用当前项目自身作为输入：

| 项目 | 结果 |
|---|---|
| 请求模型 | `ark-code-latest` |
| 服务端实际模型 | `glm-5.2` |
| 证据包 | 5 个文件、10,000 字符 |
| 输入 Token | 8,662 |
| 输出 Token | 6,803 |
| 总 Token | 15,465 |
| 延迟 | 80.288 秒 |
| Schema 修复 | 0 |
| 目标/约束/决策/Oracle | 6 / 8 / 4 / 4 |
| 证据引用 | 51/51 路径与 symbol 检查通过 |
| 最终状态 | `pending_human_review` |

这次运行证明 API、JSON、枚举约束和证据引用链可工作，不证明所有业务语义已经
正确；下一步仍需人工审核，并用新的分阶段链完成在线对照。
