# LLM 项目语义编译器

> **所属模块**：B — 语义理解与知识记忆  
> **交叉分类**：A × B — 项目证据导航与结构化语义编译  
> **文档职责**：说明从仓库证据到结构化项目语义的编译链。

> 当前阶段：单轮主链已真实验收；分阶段 Agent 链代码已实现并完成离线验收  
> 导航 Provider：默认 `MIMO_*`，也可替换为 DeepSeek 等低成本模型  
> 强分析 Provider：默认由官方 Claude Code 客户端调用 Opus-5，也可显式切换其他 Provider  
> 安全原则：LLM 负责理解，程序负责索引与核验，人工负责最终业务确认

## 0. 新的分阶段语义链

`compile-semantics-staged` 不再让一个模型拿到静态证据包后一次作答。当前实现链为：

```text
仓库文件枚举
  → 排除论文、密钥、二进制和构建产物
  → Python AST 提取符号、import、call 和代码角色先验
  → 排除训练日志、已训练权重、结果表和求解结果等运行 Artifact
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
  → 程序从 85 项目录做确定性多视图召回
  → 缺失 IR 字段标记 + 最多 20 项高召回选择包
  → 可选独立 Constraint Impact Critic
  → 返回 candidate_id 白名单校验
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
| 指标召回（无 LLM） | 程序 | family 硬门、variant 激活、mechanism/decision 排序、IR 缺口 | 已审计语义 + 85 项/1056 关系目录 | 最多 20 项高召回包 + fingerprint |
| 影响 Critic | 默认 Opus | 白名单选择、硬约束安全门与影响档位 | 已审计语义 + 问题族知识 + 紧凑候选包 | 影响报告与召回审计 |

Navigator 的文件角色明确区分：

```text
entrypoint / source / environment / model / training / evaluation
config / test / project_document / data_schema / validator / solver / other
```

其中 `evaluation` 是 benchmark、启发式比较或已训练模型评估，不等于约束测试；
`training` 也不等于部署入口。`train_log`、`trained_network`、`test_results` 和
`or_solution` 默认只作为 Artifact 排除，避免模型用训练曲线或历史结果反推项目语义。

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
| Kimi | `kimi-k3` | 仅保留对照；不再承担严格 JSON 主分析 |
| MiMo | `mimo-v2.5-pro` | `thinking.enabled` + 接口支持的最高档 `reasoning_effort=high` |
| Claude | `claude-opus-5` | 官方 Claude Code CLI、`--effort` 显式透传、无工具、无会话持久化、逐调用预算上限 |

对于推理模型，Provider 不再发送可能覆盖其推荐推理配置的 temperature。请求会同时
记录 requested/effective reasoning effort、thinking mode、可见的 reasoning 字符数
和网关返回的 thinking token。MiMo 接口只接受 `low/medium/high`，所以统一请求的
`max` 会被明确映射为该接口的最高合法值 `high`，不是静默降级。

DeepSeek、Kimi、MiMo 和火山网关已完成在线 smoke。Opus-5 通道拒绝普通 HTTP
客户端，因此不能复用 Anthropic-compatible Provider；当前改用已安装的官方
Claude Code CLI。2026-07-25 的最小 JSON 探针成功，耗时约 13.7 秒，客户端报告
成本约 0.028 美元。中转接口返回的模型名仍属于外部信任边界。

2026-07-26 调试确认，旧 Provider 虽接收 `reasoning_effort`，却没有传给 Claude Code
CLI；现已显式写入 `--effort low|medium|high|xhigh|max` 并有命令捕获回归测试。新版
最小 Opus-5 探针为 3.9 秒/36 Token；当前 FJSP Critic 使用 low 档在 66.1 秒完成。
中转长请求仍可能发生 524 或 240 秒本地超时，因此正式顺序固定为“本地测试→便宜
模型完整链→Opus 最小探针→Opus 正式复核”，失败 Trace 不自动无限重试。

### 6.1 当前模型路由决定

- MiMo 只画仓库导航，输出失败可以低成本重试；
- Opus-5 固定承担三个语义批次和最终综合；
- Opus-5 默认显式调用项目 `/scheduling-code-semantics` Skill，只开放
  `Read/Glob/Grep`，用于补充跨文件代码追踪；
- Kimi 不再作为该严格结构化总结链的默认主模型。FJSP 实测中，它在修复提示后
  仍两次擅自给固定 `batch_id` 添加轮次后缀，消耗多次长调用却无法产生合法 Artifact；
- 影响 Critic 默认使用 Opus，以降低入口阶段遗漏关键指标和诊断项的风险；DeepSeek
  暂停承担候选删除、降权或最终审核，影响审核 CLI 会直接拒绝该路由。单轮链默认 Anthropic
  协议，分阶段链默认 `--impact-protocol claude-code`。

### 6.2 调用定位与防失控

每个 Provider 都由 `LLMRunTrace` 包装。终端会实时输出：

```text
[LLM][START] call=analyst-0003 provider=semantic-analyst-opus model=claude-opus-5 task=staged_semantic_batch batch=objectives_constraints_and_paths round=0 repair=0
[LLM][DONE]  call=analyst-0003 ... elapsed=... tokens=... cost_usd=...
[LLM][ERROR] call=analyst-0003 ... error=...
```

同样的信息写入 `<output>.events.jsonl`，并额外保存 UTC 时间、响应哈希、请求次数和
可得的成本。Skill 模式还记录声明 Skill、实际工具序列和工具调用次数，但不记录
工具参数或读取正文。错误事件额外给出 `error_category` 和面向人的
`likely_cause`，区分
结构化输出、超时、路由/鉴权、模型路由、客户端兼容和网络传输。为保护项目和密钥，
日志不记录 System Prompt、User Prompt、响应正文和 API Key。所有 schema repair
也有独立 call ID，因此可以区分“模型首答卡住”与“修复调用卡住”。Opus CLI 每个
调用有 `--max-budget-usd` 硬上限，且关闭工具和会话持久化。

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

默认用 `MIMO_*` 导航、官方 Claude Code/Opus-5 做强分析：

```bash
PYTHONPATH=src .venv/bin/python -m causal_schedule_lab.cli \
  compile-semantics-staged \
  --project-root /path/to/project \
  --navigator-provider-prefix MIMO \
  --analyst-protocol claude-code \
  --max-rounds-per-batch 3 \
  --max-reads 18 \
  --claude-code-max-budget-usd 8 \
  --output outputs/staged_semantic_compilation.json
```

如需单独指定日志：

```bash
  --event-log outputs/fjsp_opus.events.jsonl
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

复用已经审核/保存的语义结果，只重跑次级指标与约束影响 Critic：

```bash
PYTHONPATH=src .venv/bin/python -m causal_schedule_lab.cli \
  review-constraint-impact \
  --semantic-compilation outputs/fjsp_test/fjsp_opus5_staged_semantics.json \
  --protocol openai \
  --provider-prefix MIMO \
  --output outputs/fjsp_test/fjsp_mimo_metric_catalog_replay.json
```

该命令先读取保存的 `SemanticAnalysis` 和 evidence packet，再执行当前 85 项目录的
确定性召回及一次 Critic。它兼容大题库接入前的旧 Artifact，不要求旧文件已经包含
`metric_recall`；不会重跑 Navigator、三批 Analyst 或代码复读。正式 Opus 复核应在
便宜模型跑通 Schema、白名单和落盘后再执行。

当前 Opus 命令示例：

```bash
PYTHONPATH=src .venv/bin/python -m causal_schedule_lab.cli \
  review-constraint-impact \
  --semantic-compilation outputs/fjsp_test/fjsp_opus5_staged_semantics.json \
  --protocol claude-code \
  --reasoning-effort low \
  --output outputs/fjsp_test/fjsp_opus5_metric_catalog_high_recall.json
```

后分析知识先验不再复用原始全文词法命中，而是根据已审计的结构化语义激活单一相关
分支。Prompt 采用紧凑 JSON；次级目标携带固定三档 `confidence`。未知扩展字段从
运行时控制数据中隔离，并把路径和值写入 `schema_extensions`，不再因此废弃整次
返回；已知字段类型、候选 ID、引用与重复性仍严格校验。

## 8. 当前限制与下一步

新增的约束影响 Critic 分别输出可行性重要度、决策杠杆、目标敏感度和候选区分度。
四个维度均从有限档位选择，程序再固定映射为排序分数；LLM 不填写自由小数。
次级指标的中文名称、计算定义和单位也由程序目录固定。程序先从 85 项目录缩到最多
20 项，Critic 以高召回原则保留所有合理相关项，只能选择候选包中的
`catalog_candidate_id` 并说明项目证据、关系、可用性和三档置信度；返回 ID、重复选择
和来源约束由程序严格校验，hard validation 由程序强制为 required 并留痕。当前默认使用
Opus 做高召回审核；DeepSeek 暂停承担候选
删除、降权或最终审核，影响审核 CLI 会直接拒绝该路由。

题库不是封闭世界。Critic 还可输出两类代码证据提案：

- `novel_secondary_targets`：题库外可测量机制，必须绑定现有 constraint 和精确
  `file::symbol`，限定为 candidate schedule / construction trajectory，并给出同实例
  候选变化依据与 intervention handle；
- `novel_diagnostic_targets`：资格/表示一致性、Oracle、Validator、下界、训练或实现
  风险等不应进入优化排序的项目特有检查。

精确重名、未知证据、未知约束或同轮重复只拒绝对应提案并保存原因，不废弃目录内
选择。所有开放提案固定为 `proposed`，不会自动进入 active 目录、Factor 或优化器。
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
- Round 3–6 指标知识统一加载与原 active 11 项兼容映射；
- family/variant/mechanism/decision 多视图确定性召回；
- required IR 缺口标记、最多 20 项高召回 Prompt、置信分层、扩展字段审计和返回 ID 白名单；
- 目录版本、候选原始数量、字符估计、批次和上下文 fingerprint 审计。

尚未实现：

- 分片模块的递归层级 Map/Reduce（当前只有一层分片 + 一次合并）；
- 批次间自动矛盾裁决（当前会记录 contradiction，不会替模型选择一方）；
- 人工 approve/reject/annotate CLI；
- 将审核结果转换成正式 `ProjectSemantics`；
- Provider 级总费用预算和熔断状态持久化；
- 各 API 阶段的即时 checkpoint 与失败后断点恢复；
- 新旧 Prompt 链与多模型的系统化在线评测集。
- 36 项诊断大库的动态召回与选择链；
- 指标召回的跨问题族人工 Recall/Precision/NDCG 校准；
- adapter 级真实 IR 字段审计（当前只使用语义证据中显式字段）。

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
