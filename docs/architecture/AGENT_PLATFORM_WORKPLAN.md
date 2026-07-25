# Agent 平台壳实施计划

> 优先级：当前最高  
> 基线版本：`0.5.0`  
> 建立日期：`2026-07-24`  
> 原则：先完成可测试、可替换、可恢复的 Agent 平台壳，再训练因果、策略或生成模型

## 1. 为什么先做平台壳

因果识别、CIP 学习、Agentic RL、条件生成和 Oracle 都会持续变化。如果先把某个
算法写死在控制器里，后续会反复改入口、状态、日志、预算和恢复逻辑。

平台壳先固定以下契约：

```text
Project
  → Runtime
  → Provider / Tool Registry
  → State Machine
  → Budget & Safety Gates
  → Artifact / Memory / Audit
  → Checkpoint / Resume / Replay
  → Algorithm Plugins
```

壳层只负责组织和约束，不声称任何学习模型已经训练完成。

## 2. API 使用原则

当前 `compile_project_semantics(provider=None)` 只能完成 AST、文档索引和规则
标签，适合作为开发 fallback，不能完成生产级的完整项目理解。

未知项目的完整语义编译必须通过 `ModelProvider` 调用具备长上下文和工具使用能力
的 LLM。标准流程为：

```text
论文与补充材料 → LLM 提取问题、环境、目标、约束和实验协议
→ 程序建立代码/文档/配置/测试索引
→ LLM 多轮检索代码并形成实现语义
→ PaperSemantics 与 CodeSemantics 逐项对齐
→ 程序核对引用、符号、配置、测试和可执行证据
→ 人工审核隐含业务规则与无法机械证明的声明
→ 生成带状态和来源的 ProjectSemantics
```

要求：

- 没有 API Key 时，只允许索引、审计已有人审语义和运行平台测试；
- 无 API 模式不得声称完成未知项目的完整语义识别；
- LLM 是生产级语义编译的必要推理组件；
- API 输出不能直接标记为 `verified`；
- API 不能绕过硬约束、预算门或 Full Oracle；
- Provider 必须记录模型、请求、Token、费用、延迟和重试；
- LLM 每项结论必须引用具体文件、符号、配置、测试或文档段落；
- 正式案例优先要求论文、代码、实例和结果能够通过 manifest 对应；
- 论文和代码冲突时必须显式输出 `conflict`，不能由 LLM 自行选择一方；
- 超长项目必须支持分区索引、检索、递归摘要和跨模块一致性复核；当前
  `semantic_agent.py` 已实现约 70k 字符分片导航、统一合并和三类语义批次；
- 原始实例、Oracle 输入和训练数据不能因上下文压缩而失真；
- 密钥只从环境变量或秘密管理器读取，不写入配置、日志或 Artifact。

当前语义专用工具治理链：

```text
LLM EvidenceReadRequest
  → 证据充分性门
  → 项目路径白名单
  → 论文/密钥/二进制排除
  → AST symbol 存在门
  → call/import/config/validator 关系门
  → 去重与每轮/总量预算
  → 精确函数读取
  → ToolEvidence + approve/reject 审计
```

该工具已经可执行，但只服务语义 Agent，不等同于 S2 计划中的全平台工具注册表。

## 3. 当前壳层覆盖情况

| 能力 | 当前证据 | 状态 | 缺口 |
|---|---|---|---|
| 项目清单与上下文 | `project.py` | 主闭环已运行 | 缺运行级上下文、版本和预算 |
| 项目适配器 | `SchedulingProjectAdapter` | 主闭环已运行 | 缺 capability 声明和生命周期 |
| Generator/Oracle 插件 | `plugins.py` | 主闭环已运行 | 不是统一工具注册表 |
| 核心数据契约 | `models.py` | 主闭环已运行 | 运行状态、事件和 Artifact 契约不足 |
| Agent 动作空间 | `agentic_rl.py` | 代码已实现但部分接入 | 未统一纳入状态机 |
| 控制器 | `controller.py` | 主闭环已运行 | 过程写死，缺显式状态和恢复 |
| CLI | `cli.py` | 主闭环已运行 | 缺统一 run/resume/replay/status |
| 审计日志 | `audit.py` | 主闭环已运行 | 仅 JSONL，无事件版本和索引 |
| 训练状态 | `training_pipeline.py` | 代码已实现但未配齐 runner | 只覆盖训练，不覆盖在线运行 |
| LLM 单轮语义编译接口 | `llm_semantics.py`、`providers/` | 火山/GLM-5.2 单轮主链已实测 | 保留为回归基线；缺人工审核和预算持久化 |
| 分阶段语义 Agent | `semantic_agent.py` | 分片 Navigator、三批 Analyst、证据充分性 MCQ、AST 复读门和短期记忆代码已实现并离线验收 | 尚未正式在线质量对照；未接统一 runtime/checkpoint |
| 调度问题族 RAG | `semantic_knowledge.py`、`knowledge/scheduling_families.json` | JSP/FSP/HFSP/FJSP 与八类工程模式检索已进入语义编译 | 缺多案例覆盖率与知识版本审批 |
| 图谱长期记忆 | `storage/graph_store.py` | SQLite 节点、边、别名、证据、干预与版本代码已实现 | 尚未进入默认 runtime；缺统一人工审批门 |
| 分层证据检索 | `storage/memory.py` | L0–L4 FTS5、图门控、种子迁移和 PDF/文本 proposed 导入已实现 | 尚未进入默认语义编译和 Artifact 管理 |
| 约束影响 Critic | `constraint_impact.py` | 枚举选择题、固定分数映射、可选 CLI 与安全门已实现 | 尚未在线验收，未进入 CIP/Agent |
| 论文—代码盲测 Harness | `semantic_harness.py` | 网上 L2D 案例已真实运行 | 仅一个案例；已建立逐项人工核验表，论文标签仍未确认 |
| 多 Provider 语义 benchmark | `openai_compatible.py`、`anthropic_compatible.py`、`semantic_harness.py` | 五模型历史基线已运行；最高质量协议已接入 | MiMo-Pro 等待人工真值后正式重跑；Opus adaptive/max 尚待中转恢复后在线确认 |
| 反思隔离 | `reflection.py` | 代码已实现但未接入 | 无真实 LLM 调用和回放流程 |
| Artifact 管理 | CLI 直接写文件 | 接口不足 | 无统一目录、manifest、hash |
| 在线 Checkpoint | 无 | 尚未实现 | 无 crash-safe resume |
| 运行重放 | 有 schedule hash | 尚未形成系统 | 缺完整输入、版本和事件重放 |
| 人工审批门 | 无统一接口 | 尚未实现 | 缺 approve/reject/annotate |
| 时间/Token/费用预算 | 候选和 Oracle 计数 | 部分实现 | 无统一 Budget Manager |
| 重试、退避和熔断 | 无 | 尚未实现 | API/Oracle 故障不可控 |
| 可观测性 | JSON 输出 | 部分实现 | 缺结构化 event、metric、trace |

## 4. 目标目录边界

计划增加或稳定以下平台层，算法模块通过协议接入：

```text
src/causal_schedule_lab/
├── runtime/
│   ├── context.py          # RunContext、版本、seed、预算、路径
│   ├── state.py            # 显式状态机与合法迁移
│   ├── events.py           # 版本化事件协议
│   ├── engine.py           # 运行编排，不包含具体调度算法
│   ├── checkpoint.py       # 原子保存、恢复与迁移
│   └── replay.py           # 确定性重放
├── providers/
│   ├── base.py             # ModelProvider 协议
│   ├── rule.py             # 仅供索引、测试和已知语义 fallback
│   ├── semantic_agent.py   # 多轮项目理解、证据引用和结构化输出
│   └── registry.py         # 延迟加载具体 API Provider
├── tools/
│   ├── base.py             # ToolSpec、ToolResult、Capability
│   ├── registry.py         # graph/CIP/solver/oracle 等注册
│   └── executor.py         # timeout、retry、circuit breaker
├── storage/
│   ├── artifacts.py        # hash、manifest、原子写入
│   ├── audit.py            # append-only event log
│   ├── memory.py           # 已实现：L0–L4 分层证据、FTS5 与迁移
│   └── graph_store.py      # 已实现：SQLite 属性图、版本、作用域与干预
├── budgets/
│   ├── manager.py          # 时间、Token、费用、候选和 Oracle
│   └── policy.py           # 预检查和超预算动作
└── approvals/
    └── gates.py            # 人工审核与外部责任
```

目录可以在实现时微调，但职责不能重新堆回一个控制器文件。

## 5. S0–S8 实施顺序

### S0：平台契约冻结

定义：

- `RunContext`；
- `RunState`；
- `AgentEvent`；
- `ArtifactRef`；
- `BudgetSnapshot`；
- `ToolSpec/ToolResult`；
- `ModelProvider`；
- `ApprovalRequest/Decision`。

完成门：数据结构有版本号、序列化测试和向后兼容策略。

### S1：显式状态机

建议状态：

```text
CREATED
→ LOADING_PROJECT
→ SEMANTIC_AUDIT
→ READY
→ DIAGNOSING
→ PROPOSING
→ GENERATING
→ VALIDATING
→ ACCEPTING / REJECTING
→ CHECKPOINTING
→ PAUSED / COMPLETED / FAILED / CANCELLED
```

完成门：非法迁移被拒绝；每次迁移生成事件；异常不会留下伪完成状态。

### S2：统一工具与 Provider 注册表

将 adapter、graph、CIP、operator、generator、solver、validator、Oracle 和
LLM Provider 统一声明为 capability。

完成门：

- 工具可发现、可替换；
- 缺失可选能力时有明确 fallback；
- 导入插件不自动发起 API 或外部写操作。

### S3：预算、安全与外部调用治理

统一管理：

- wall-clock；
- Token；
- API 费用；
- 候选数；
- solver 时间；
- Light/Full Oracle 次数；
- 最大重试次数。

完成门：调用前预检查、调用后记账；超预算进入可解释状态；密钥不落盘。

### S4：Artifact、事件日志和可观测性

每次 run 建立独立目录，保存：

- resolved manifest；
- 输入 hash；
- 环境和依赖版本；
- append-only events；
- metrics；
- checkpoints；
- candidates；
- Oracle 结果；
- 最终报告。

完成门：所有结果能追溯到 run、输入、工具版本和事件。

### S5：Checkpoint、Resume 与 Replay

完成门：

- 在关键阶段原子 checkpoint；
- 进程中断后不会重复已完成的高成本 Oracle；
- `resume` 从最后合法状态继续；
- `replay` 在相同 seed、版本和输入下重现动作与 hash；
- 版本不兼容时拒绝静默恢复。

### S6：人工审批门

支持：

- 语义 `unknown` 审核；
- 高成本 Oracle 调用批准；
- 数据与许可证确认；
- 发布或业务验收；
- approve/reject/annotate/defer。

完成门：外部责任不能被 Agent 自动伪造为通过。

### S7：统一 CLI 和配置

目标命令：

```text
inspect
run
status
pause
resume
cancel
replay
validate
artifacts
```

完成门：CLI、Python API 和未来服务接口调用同一个 runtime engine。

### S8：空壳端到端验收

先使用 fake/rule provider 验收平台机制，再以至少一个真实 LLM Provider 验收
未知项目语义编译：

- 正常执行；
- 暂停/恢复；
- API 缺失时降级到“只索引/只审计”，并明确拒绝完整语义声明；
- Tool timeout/retry/circuit breaker；
- 预算耗尽；
- 人工拒绝；
- crash recovery；
- deterministic replay。

完成门：

- 平台机制可以在不训练模型的情况下完整测试；
- 真实 LLM Provider 可以完成一次受预算约束的项目语义编译；
- 无 LLM 时系统不会把规则索引误报为完整项目理解。

## 6. 壳完成后再接入的模块

依次接入：

1. 条件化因果核心点 C0–C6；
2. 反事实数据采集；
3. CIP 多任务模型；
4. 条件生成器；
5. Agentic RL；
6. 多问题族实验；
7. LLM reflection 和规则蒸馏。

接入某模块时只新增插件和状态处理，不重新修改平台基本契约。

## 7. 用户与 Codex 分工

### Codex

- 实现平台数据契约、状态机、注册表和执行引擎；
- 编写 fallback、预算、存储、恢复、重放和测试；
- 保持 Provider 可替换，同时将 LLM 设为未知项目完整语义编译的必需能力；
- 更新架构、公式矩阵和实验记录。

### 用户

- 确定允许使用的 API、模型和预算；
- 提供密钥时只放在本地环境变量；
- 审核业务语义和人工审批门；
- 确定数据、模型和输出的发布边界；
- 对最终业务结果做验收。

## 8. 当前决策

从 `2026-07-24` 起：

- Agent 平台壳 S0–S8 为最高实施优先级；
- 因果模块 C0–C6 保留为下游计划；
- 在平台壳验收前，不开始大规模反事实采集和模型训练；
- 可以保留当前规则闭环作为壳层测试用 deterministic fallback。
- 论文只作为训练阶段特权信息；生产语义 Agent 必须支持 code-only 和碎片文本输入。
- 长期记忆采用 SQLite 图谱事实主干与 L0–L4 分层证据索引；向量检索只可在已选
  子树内重排，不能写入事实或替代证据审核；
- 正式目标与机制目标分离；参数只有在同实例候选间发生变化、且具备直接或经干预
  验证的间接可控性时，才可进入优化优先级；
- 具体契约见
  [长期记忆、分层检索与因果机制目标](LONG_TERM_MEMORY_AND_CAUSAL_MECHANISMS.md)。
