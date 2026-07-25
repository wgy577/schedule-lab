# 项目状态与持续路线图

> 文档性质：长期维护的状态总表  
> 当前版本：`0.5.0`  
> 最近更新：2026-07-25  
> 更新责任：每次代码、知识、实验、训练、Oracle 或发布状态变化时同步维护。

## 1. 当前里程碑

| 里程碑 | 状态 | 可验证证据 | 下一门槛 |
|---|---|---|---|
| 通用 JSP/FSP/FJSP/HFSP IR | 主闭环已运行 | 四类 manifest 与端到端测试 | 更多公开实例 importer |
| LLM 项目语义编译 | 主闭环已运行 | 真实 Provider Artifact 与证据审计 | 分阶段链在线对照和人工落库 |
| 分阶段语义 Agent | 代码已实现但未接入 | Navigator 分片、三批 Analyst、AST 复读门、短期记忆和离线测试 | 用已确认标签运行多模型在线对照 |
| 经典问题族与工程模式知识 | 主闭环已运行 | JSON seed 与检索测试 | 新综述人工审核 |
| SQLite 图谱长期记忆 | 代码已实现但未接入 | 73 节点、85 边迁移 smoke；幂等测试 | 接入语义编译默认检索 |
| L0–L4 分层证据检索 | 代码已实现但未接入 | 72 证据块、FTS5 查询和 PDF/文本导入测试 | 冲突审核 UI/CLI 与 embedding 可选重排 |
| makespan 机制目标库 | 代码已实现但未接入 | 8 个定义、8 个计算器入口 | 项目语义路由与跨族校准 |
| 候选变化与可控性资格门 | 代码已实现但未接入 | constant、hypothesized、verified 单测 | 真实 replay 生成控制证据 |
| 干预记录与效应后验 | 代码已实现但未接入 | SQLite 去重、作用域查询、保守后验单测 | Controller 自动写入与预算采集 |
| CIP/局部修复/四级验证 | 主闭环已运行 | 现有端到端测试 | 学习式 CIP 与更多领域 Oracle |
| 跨问题族效果结论 | 尚未实现 | 无真实完整实验矩阵 | 公开基准、消融与统计 |

## 2. 当前长期记忆实物

生成命令：

```bash
causal-schedule-lab memory-build
```

默认位置：

```text
outputs/memory/schedule_memory.sqlite3
```

当前种子规模：

```text
73 nodes
85 edges
310 unique aliases
72 evidence chunks
0 real intervention experiments
```

数据库是可重建 Artifact，不作为人工编辑源。人工维护源仍是：

- `knowledge/scheduling_families.json`；
- `knowledge/mechanism_targets.json`；
- 后续审核通过的知识清单和外部证据；
- 不可变实验记录。

## 3. 您后续需要提供的信息

优先寻找当前知识种子未充分覆盖的综述、工程论文或高质量说明：

| 优先级 | 信息缺口 | 希望从资料中确认 |
|---|---|---|
| P0 | 各问题族机制与 makespan 的关系 | 哪些中间指标在什么条件下具有稳定解释力 |
| P0 | 工业约束交互 | 运输×机器、人员×机器、维护×排序、buffer×blocking |
| P0 | 常见局部算子与适用条件 | 算子直接控制什么变量、会释放多大传播闭包 |
| P1 | 动态/随机调度 | 状态变化、重调度触发、稳定性与鲁棒性 |
| P1 | 多保真验证 | 代理误差、仿真成本和 Full Oracle 选择 |
| P1 | 因果/反事实调度研究 | 干预单位、控制组、识别假设和失败边界 |
| P2 | 规模与迁移 | 跨实例/跨问题族可以共享哪些后验 |

资料可以是 PDF、Markdown 或纯文本。第一步只进入 `proposed` 证据层：

```bash
causal-schedule-lab memory-ingest-document \
  --file /path/to/review.pdf \
  --title "Review title" \
  --source-kind peer_reviewed_survey \
  --scope FJSP
```

导入不等于采信。随后需要执行：

```text
提取候选声明
→ 与现有图谱去重
→ 检查来源和冲突
→ 代码/论文/实验分层
→ 人工审核
→ active 或 rejected
```

## 4. 下一阶段实施顺序

### P0：在线验收分阶段语义链

- 用低成本 Navigator 与高能力 Analyst 分开运行；
- 比较旧单轮链和新分阶段链的约束召回、错误引用、Token 与人工修正量；
- 测试 `sufficient/partial/insufficient/contradictory` 的读请求校准；
- 暂不以单个 smoke 宣称质量提升。

### P0：把已实现记忆接入语义链

- 分阶段链已可选执行图邻域和 L0–L4 检索；下一步验证召回质量后再进入默认 runtime；
- LLM 输出中记录使用过的 node/chunk ID；
- 提案只能进入 `proposed`；
- 人工审核后才写 `active` 关系。

### P0：建立真实机制干预数据

- 在同一 instance/incumbent/state 下生成受控局部候选；
- 记录因子变化、机制变化、传播闭包和最终目标变化；
- Full Oracle 失败仍写入记忆；
- `verified_indirect` 必须来自 replay，而不是 LLM 判断。

### P1：后验进入候选排序

- 先离线 shadow mode，只输出建议不控制搜索；
- 比较无记忆、LLM 先验、经验后验三种排序；
- 校准有效率、保守收益、时间和 Token 成本；
- 通过后才允许影响 CIP/Agent 的 attention。

### P1：跨问题族验证

- JSP/FSP/FJSP/HFSP 分层实验；
- leave-one-family-out；
- 固定实例划分，禁止训练/测试泄漏；
- 报告未改善、不可行和 Oracle 失败，而不只报告最好结果。

## 5. 尚未解决的设计决策

| 决策 | 当前默认 | 需要什么证据后再修改 |
|---|---|---|
| 间接可控门槛 | 至少 1 次 replay 且一致率 ≥ 0.8 | 多实例重放稳定性 |
| 图数据库 | SQLite | 单库规模、并发和查询延迟达到迁移阈值 |
| 向量检索 | 不作为事实源；尚未接入 | 证明在同子树内提高召回且不增加错族 |
| 后验模型 | Beta 有效率 + 保守收益统计 | 足够干预样本后比较分层贝叶斯模型 |
| 三元交互 | 默认不测 | 成对模型持续出现大残差 |
| 跨族迁移 | 默认隔离 | leave-one-family-out 有统计证据 |

## 6. 每次更新必须维护的文件

| 变化 | 必须同步 |
|---|---|
| 大模块或数据流 | `PROJECT_MODULE_GRAPH.md`、`docs/architecture/SYSTEM_ARCHITECTURE.md` |
| 模块图、流程图、截图或其他当前项目图片 | `PROJECT_MODULE_GRAPH.md`、`PROJECT_MODULE_GRAPH_INTERACTIVE.html`、对应展示入口；同步更新或明确退役 |
| 当前状态或路线 | 本文件 |
| 长期记忆/机制 | `docs/architecture/LONG_TERM_MEMORY_AND_CAUSAL_MECHANISMS.md` |
| 公式或统计 | `docs/architecture/FORMULA_IMPLEMENTATION_MATRIX.md` |
| 因果数据/CIP | `docs/architecture/CAUSAL_MODULE_WORKPLAN.md` |
| 平台/存储/Provider | `docs/architecture/AGENT_PLATFORM_WORKPLAN.md` |
| 已执行实验 | `EXPERIMENTS.md` |
| 用户入口 | `README.md` |

## 7. 状态更新模板

```markdown
### YYYY-MM-DD / change-id

- 变化：
- 代码：
- 状态从：
- 状态到：
- 验证：
- 产物：
- 仍未完成：
- 下一步：
```

只有实际测试、运行路径和 Artifact 能推动状态升级。路线图或 Prompt 不能证明实现。

## 8. 最近维护记录

### 2026-07-25 / docs-layout-v1

- 变化：将状态路线图和两层项目图移动到仓库根目录；
- 分类：建立 `docs/architecture`、`docs/semantics`、`docs/experiments`、
  `docs/guides`；
- 同步：README、维护 Skill、审计脚本和全部 Markdown 相对链接；
- 验证：文档链接检查、架构审计和项目测试；
- 后续：新增文档必须先选择所属一级分类，不再直接堆放到 `docs/` 根目录。

### 2026-07-25 / module-graph-interactive-v1

- 变化：新增根目录可缩放项目图；
- 交互：第一层/第二层切换、滚轮缩放、按钮缩放、拖拽、适应窗口、重置、全屏；
- 约束：交互图与 `PROJECT_MODULE_GRAPH.md` 使用同一两层结构，模块变化时同步更新；
- 验证：浏览器布局与交互检查。

### 2026-07-25 / visual-sync-policy-v1

- 变化：Markdown 不再嵌入静态 PNG 项目图；
- 规则：模块、状态、数据流、实现路径或里程碑变化时，同步更新 Markdown Mermaid、
  交互图和仍作为当前版本使用的其他图片；
- 退役：无法同步的旧截图或导出图必须标记为历史版本或移除；
- 验证：维护 Skill 和架构审计加入图形同步检查。

### 2026-07-25 / staged-semantic-agent-v1

- 变化：新增低成本分片 Navigator、高能力三批 Analyst、证据充分性 MCQ、
  受控 AST 复读工具、压缩短期记忆和最终综合；
- 安全：模型只能申请读取，程序按路径、关系、symbol、重复和预算确定性审批；
- 记忆：SQLite 图谱/FTS 检索可作为批次先验，但禁止作为当前项目 evidence；
- 状态：代码已实现但未接入默认 runtime；尚未执行正式在线质量对照；
- 验证：离线 Fake Provider 端到端、复读审批和秘密/论文排除测试。
