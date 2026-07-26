# 调度语义知识库与优化影响权重

> **所属模块**：B — 语义理解与知识记忆  
> **关联模块**：D — 诊断与因果机制  
> **交叉分类**：B × D — 问题族知识、约束影响与诊断先验  
> **文档职责**：维护问题族知识、检索规则和影响先验的证据边界。

> 状态：问题族检索与 85 项指标确定性召回已接入；独立影响 Critic 可选调用，尚未
> 形成多案例校准结果，proposed 指标尚未进入优化器  
> 建立日期：`2026-07-25`

## 1. 目的

语义 Agent 不应在每个项目上重新“发现”经典 JSP 的前置、单机互斥和不可抢占，
也不应把论文对比算法、硬件或 benchmark 当成调度约束。新的处理顺序是：

```text
项目代码/论文/说明
        │
        ▼
精确别名 + 词法检索
        │
        ├── JSP / FSP / HFSP / FJSP 经典档案
        └── 常见变体触发项
        │
        ▼
LLM 只核对项目与经典档案的相同点和增量差异
        │
        ▼
独立结构化选择 Critic 做情境化影响评分
        │
        ├── 可行性重要度
        ├── 决策杠杆
        ├── 目标敏感度
        └── 候选区分度
        │
        ▼
人工确认 → 进入诊断/优化注意力
```

知识库是先验，不是项目证据。项目是否真的具有某个变体，仍必须引用代码、配置、
测试、实例或正式文档。

## 2. 当前知识库

文件：

`src/causal_schedule_lab/knowledge/scheduling_families.json`

首批覆盖：

| 问题族 | 经典识别核心 | 主要优化决策 |
|---|---|---|
| JSP/JSSP | 每个作业有自己的机器访问顺序；经典情况下每道工序绑定指定机器 | 同机排序、逐工序选择 |
| FSP | 所有作业共享同一机器/阶段顺序；区分普通 FSP 与 PFSP | 各机排序；PFSP 为单一全局排列 |
| HFSP | 所有作业经过同一阶段顺序，至少一个阶段有并行机器 | 阶段机器分配 + 同机排序 |
| FJSP | 每道工序有候选机器/模式，机器选择可能改变加工时间 | 机器分配 + 同机排序 |

已存常见变体：

- JSP：release/dynamic、setup、transport/AGV、no-wait/blocking、双资源、
  stochastic/robust；
- FSP：PFSP、NPFSP、no-wait/blocking、setup、distributed、reentrant；
- HFSP：异构并行机/eligibility、blocking/buffer、setup、no-wait、batch、
  transport；
- FJSP：部分/完全柔性、dynamic、distributed、transport/AGV、双资源、
  setup/maintenance、energy/multi-objective、stochastic/robust。

### 2.1 工程约束模式

问题族变体之外，知识库现在单独保存跨问题族复用的工程模式：

| 工程模式 | 新增实体/决策 | 典型验证 |
|---|---|---|
| 移动运输/AGV | 车辆、运输任务、路径、充电；车辆分配与路由 | 运输连续性、容量、避碰、轨迹 Oracle |
| 人员双资源 | 人员、技能、班次、疲劳；机器—人员联合分配 | 技能、日历、绑定、人体工学模型 |
| 维护与机器可用性 | 维护任务、故障、健康状态；维护时机与重调度 | 日历、可靠性模型、仿真 |
| 动态事件与重调度 | 事件流、冻结区、滚动窗口；触发和重调度范围 | 状态回放、稳定性、数字孪生 |
| 缓冲/物料/齐套 | 缓冲、WIP、物料套件；释放与同步 | blocking、buffer occupancy、离散事件仿真 |
| 顺序相关换型 | 作业族、切换矩阵、清洗规则；序列选择 | setup matrix validator |
| 能耗与可持续性 | 功率、分时电价、碳因子；速度/模式与多目标权衡 | 能耗计算器、设备状态模型 |
| 数字孪生/实时接口 | MES/IoT 状态、版本与延迟；在线触发 | 状态一致性、仿真或权威外部 Oracle |

每个工程模式均保存：

- 适用问题族；
- 触发别名；
- 新增实体；
- 新增直接决策；
- 可选约束、目标和 Oracle；
- 给 LLM 的有限 screening questions；
- 综述或工程论文来源。

工程模式命中只能生成“待核对候选”，不能证明项目具备该约束。

种子来源包括：

- Google OR-Tools 官方 JSP 定义；
- 2022 JSSP 类型与模型综述；
- 2024 FJSP 综述；
- integrated FJSP 综述；
- PFSP/NPFSP 综述；
- HFSP 分类综述。

每个 profile 保存来源 URL、经典约束、默认假设、直接决策、变体触发词以及典型
优化影响说明。`typical_optimization_leverage` 只能用于检索排序，不能直接覆盖
当前项目权重。

## 3. 检索方式

实现：

`src/causal_schedule_lab/semantic_knowledge.py`

第一版刻意采用可解释的轻量检索：

1. 已知 family hint 时精确规范化，`JSSP → JSP`；
2. 对 JSP/FSP/HFSP/FJSP 的英文、缩写和中文别名做边界匹配；
3. 语义抽取前允许以变体触发词作导航先验；
4. LLM 完成初次分类后，最终 Critic 改用结构化 constraint/environment/objective
   激活变体和工程模式；
5. 只返回被结构化语义命中的分支，不注入全部兄弟变体。

短缩写使用 token boundary，避免 `JSP` 错误命中 `FJSP`。当前不引入向量数据库，
原因是首批知识规模小、术语明确，精确检索更稳定、便宜且容易审计。后续条目扩大后
可在该层后增加 BM25/embedding reranker，但不替换 exact family gate。

## 4. LLM 语义编译如何使用知识库

`compile_project_semantics_with_llm()` 现在默认执行知识检索，并把命中 profile 放入
Prompt。系统提示明确要求：

- 知识库不能作为项目 evidence；
- 经典约束只需核对，不必从零反复推理；
- 主要注意项目新增、删除或修改的变体；
- 对比算法、benchmark、硬件与性能主张不得列为调度硬约束。

输出 `LLMSemanticCompilation.knowledge_hits` 保存最终命中结果，便于审核为什么给模型
提供了某个先验。

## 5. 约束的“重要”必须拆成四个维度

不能只给一个含糊总权重。LLM 先对四个维度做选择题：

```text
none / very_low / low / medium / high / critical / unknown
```

| 维度 | 含义 | 典型问题 |
|---|---|---|
| `feasibility_criticality` | 违反后是否使排程不可行 | 是否必须进入 validator/Oracle |
| `decision_leverage` | 优化器能否通过修改决策利用它改善结果 | 是否值得设计算子和邻域 |
| `objective_sensitivity` | 该量变化时当前目标是否显著变化 | 对 makespan/tardiness/energy 的影响 |
| `candidate_discrimination` | 在同一实例的候选之间能否区分优劣 | 是否值得进入当前候选排序 |

程序再固定映射为：

```text
none=0.0, very_low=0.1, low=0.25,
medium=0.5, high=0.75, critical=1.0, unknown=null
```

数字只用于后续排序，原始档位始终保留。LLM 不允许自行填写小数。

还要从固定枚举输出：

- `role`：feasibility guard、decision lever、objective driver、state parameter、
  fixed instance structure、evaluation only 或 implementation only；
- `context_scope`：经典问题族、常见变体、项目特有或实验上下文；
- `controllability`：direct、indirect、fixed、external 或 unknown；
- `variation_across_candidates`：constant、decision-dependent、exogenous、mixed
  或 unknown；
- `validation`：required、not-required 或 pending-review；
- `optimization_attention` 与 `diagnosis_attention`：required、consider、
  deprioritize、exclude 或 pending-review；
- 权重理由、知识引用、置信度和待确认问题。

### 5.1 权重还必须注意的因素

1. **目标条件化**：同一约束对 makespan、tardiness、energy 的作用可能完全不同。
2. **候选条件化**：参数即使影响绝对完工时间，如果所有候选都相同，就没有候选
   区分度。
3. **可控性**：只有 direct/indirect 可控项才可能成为算子杠杆；fixed/external
   更适合作为状态或验证条件。
4. **是否紧约束**：有大量 slack 的约束可能暂时低影响；当前正 binding 的约束应
   提高诊断优先级。
5. **交互效应**：release time 单独看可能低，但与 due date、动态到达、机器故障
   联合时可能高影响，不能只做逐条独立评分。
6. **传播范围**：局部改动是否沿 precedence/resource/transport 路径扩散，决定
   closure 大小和验证成本。
7. **阶段位置**：瓶颈阶段、sink stage 和关键路径上的同类约束通常比非关键位置
   更值得诊断。
8. **证据强度**：显式 solver/validator 代码、测试、实例数据和文字说明应分别记录，
   不能只凭变量名提高权重。
9. **时间变化**：动态系统的权重会随事件和 incumbent 改变，不能永久固化。
10. **验证成本**：高影响但只能由昂贵 Full Oracle 判断的因素，需要与调用预算联合
    决策。
11. **不确定性**：`unknown` 必须保留为 null，不能为了排序强行填中间值。
12. **人工可推翻**：LLM 权重只生成建议，领域审核或反事实实验可以覆盖。

### 5.2 后续用数据校准，而不是继续增加 Prompt 自信

选择题档位是第一版先验。获得候选实验后，应该用实际数据更新：

- 约束是否 binding、slack 多大；
- 局部干预是否改变目标；
- 同约束不同位置的平均改善和失败率；
- interaction/pairwise 干预结果；
- Light 与 Full Oracle 的偏差；
- 模型档位与真实改善之间的校准曲线。

在没有这些数据前，`critical/high` 只表示 Critic 的审核优先级，不表示已证明的
因果效应。

长期记忆、分层检索、候选变化硬门以及
`operator → factor → mechanism → objective` 的后续实现契约，见
[长期记忆、分层检索与因果机制目标](LONG_TERM_MEMORY_AND_CAUSAL_MECHANISMS.md)。

### 5.3 Round 3–6 多视图候选知识与 Token 有界召回

问题族与变体知识树之外，项目现在还保存一套 proposed 次级指标/诊断知识：

- 85 个 canonical candidate 只保存一次实体正文；
- 1056 条去重关系分别连接 problem family、variant head、mechanism、decision、role、
  lifecycle 和 evidence 视图；
- 同一候选可在多个视图中出现，但通过 `candidate_id` 去重；
- no-wait/no-idle 等上下文允许改变候选角色和方向，不改 canonical 身份；
- `required_ir_fields` 从语义相关性中分离，只作为 computable channel 的硬门；
- 36 个 diagnostic 保持独立目录，不与指标身份混排；
- 原 active 11 项通过显式兼容映射全部存在于统一目录，近似但非等价者不会强行合并。

加载与召回接口为 `SecondaryMetricKnowledgeBase`。影响 Critic 被启用时，程序先从
`SemanticAnalysis` 构造问题族、显式变体、机制、可修改决策和可见 IR 字段上下文；
family 是硬门，required variant 是激活门，其余视图用于确定性排序。缺少 IR 字段只会
标成 `needs_adapter_or_evidence`，不会由模型猜测为已计算。

召回结果最多保留 20 项，并以紧凑 JSON 进入同一次 Critic 请求。LLM 采用高召回
策略：合理相关但证据较弱的候选用 `confidence=low` 保留，而不是在入口删除。LLM
只能返回这些 `candidate_id`，程序随后执行白名单、重复 ID 和来源约束校验；硬约束
验证由程序强制并在 `safety_overrides` 留痕。未知扩展字段不会控制运行时，但其路径和
值保存在 `schema_extensions`。输出中保存目录
版本、原始合格数、候选包、批次、字符估计和上下文 fingerprint，因此 Token 与路由均
可审计。题库从 55 扩展到 85 不会把全部正文注入 Prompt。

`confidence` 是模型对“候选与当前项目相关”的认识置信层，可作为后续排序的软权重，
但不是候选对 makespan 的效应大小，也不能替代变化测试、干预或 Oracle 验证。

### 开放世界代码提案

目录召回之后保留一条受控旁路。代码出现目录无法表达的新机制时，模型可提出新次级
目标；代码出现实现/参照/验证风险时，可提出新诊断。两类都必须引用当前项目已有的
constraint ID 与逐字 `file::symbol`。新次级目标额外必须满足：

1. 测量范围是候选排程或构造轨迹，不是固定实例属性；
2. 明确同一实例不同候选为什么变化；
3. 指出能够改变它的排程或解码干预；
4. 给出计算定义、单位、方向、目录缺口理由和置信度。

程序执行精确重名、同轮重复、约束引用和代码证据门。失败只进入
`novel_candidate_rejections`，不拖垮整份报告；通过项仍仅为 `proposed`。公式语义
等价、望远镜求和、代理指标重复等目前仍需人工或后续符号审核。

2026-07-26 的 FJSP 对照发现，摘要中的“未见 setup、运输、双资源”曾被词法检索
误当成正向变体。当前后分析路径统一使用 `retrieve_conditioned_on_analysis()`：
经典 FJSP 不注入 setup/dual-resource 先验，只有显式结构化约束才激活对应分支。
知识上下文、项目上下文、模板和 Schema 同时使用紧凑 JSON，Critic Prompt 约从
25.5k 字符降到 22.3k 字符。

次级目标允许返回 `high/medium/low` 三档置信度。程序将所有未知扩展字段从控制
Schema 中隔离，并审计其路径和值；扩展字段不会再让整份报告失败。未知候选 ID、重复
ID、未知来源约束和已知字段类型错误仍严格拒绝。硬约束验证状态由程序所有，模型若
返回其他值会被覆盖为 required 并写入安全审计。

## 6. L2D 中 release time 的正确处理

L2D 是标准静态 JSSP。论文在完成时间下界公式中写了首工序 release time，而公开
代码把首工序 ready time 固定为 0。

当前框架不再把它当成值得花大量推理预算的“重大论文—代码冲突”，而是：

```text
事实记录：
  代码实现所有作业 release=0

当前项目权重：
  feasibility_criticality：保留，由实例定义决定
  decision_leverage：接近 0，因为策略不能修改
  candidate_discrimination：0，因为所有候选共享相同零释放时间
  optimization_attention：关闭

条件性反例：
  如果项目是动态 JSP、作业分批到达或目标包含延期/响应，则权重必须重新评估
```

同理：

- OR-Tools、SPT/MWKR 等属于实验 comparison，不进入调度约束权重；
- 训练硬件和运行时属于实验成本；
- “选择下一工序后放到最早可行位置”属于经典构造规则，开始时间是派生结果；
- 只有能够改变候选排序、机器分配、传播闭包或目标值差异的因素才进入优化注意力。

## 7. 安全门

实现：

`src/causal_schedule_lab/constraint_impact.py`

程序强制：

1. 每条输入 constraint 必须且只能返回一次；
2. 输入中 `hard=true` 的约束必须 `must_validate=true`；
3. 低 `decision_leverage` 只会降低优化注意力，不能删除可行性检查；
4. 输出保持 `pending_human_review`；
5. Critic 默认使用 Opus 高召回审核；DeepSeek 暂停承担候选删除、降权和最终审核，影响审核 CLI 会拒绝该路由；指标定义由程序固定；
6. Critic 与初始语义抽取可以使用不同 Provider。

CLI：

```bash
PYTHONPATH=src .venv/bin/python -m causal_schedule_lab.cli \
  compile-semantics-llm \
  --project-root /path/to/project \
  --env-file .env \
  --provider-prefix SEED \
  --assess-constraint-impact \
  --impact-protocol anthropic \
  --output outputs/project_semantics_with_impact.json
```

历史上曾使用以下 OpenAI-compatible Critic 命令；当前影响审核门会拒绝 DeepSeek，
该命令只保留为历史记录，不应执行：

```bash
--impact-protocol openai --impact-provider-prefix DEEPSEEK
```

## 8. 当前状态与下一步

| 能力 | 状态 | 证据 | 缺口 |
|---|---|---|---|
| 四问题族与工程模式知识种子 | 主闭环已运行 | JSON Schema 解析、family/variant/工程模式检索测试 | 需领域专家持续审订 |
| 精确 family/variant 检索 | 主闭环已运行 | JSP/FJSP 消歧和变体命中测试 | 尚无 BM25/embedding |
| 检索先验注入语义编译 | 主闭环已运行 | 默认调用路径和测试 | 需多案例衡量 Token 节约 |
| 独立影响 Critic + 指标白名单选择 | 代码已实现但未接入默认命令，可选调用已贯通 | 当前 MIMO/Opus 同证据在线 Artifact、结构化变体先验、Schema、返回 ID 白名单、硬约束门和离线测试 | 尚无人工真值、候选扰动与 Oracle 校准 |
| 85 项/36 诊断统一目录 | 代码已实现但未进入优化器 | Round 3–6 导入、1056 条关系、11 项兼容审计 | 动态诊断选择与 active 晋级尚未完成 |
| 自动 literature refresh | 尚未实现 | 无 | 需要来源白名单、版本和人工合并门 |
| 权重校准 | 尚未实现 | 无多案例人工标签 | 需不同目标/变体的标注集 |
| 权重进入 CIP/Agent | 尚未实现 | 当前控制器未消费 | 人工确认后再接入，避免错误过滤 |
| 图谱长期记忆与分层证据检索 | 代码已实现但未接入 | SQLite Schema、幂等迁移、FTS5、外部文档导入和作用域测试 | 需进入默认语义链与人工审核流 |
| 因果机制目标发现与后验校准 | 代码已实现但未接入 | 八类计算器、资格门、干预存储和保守后验测试 | 需真实干预、跨问题族校准和 Agent 接入 |

下一阶段不应先追求更复杂的向量 RAG，而应：

1. 将已经实现的图谱与分层 evidence pack 接入语义编译 shadow mode；
2. 用 JSP/FSP/HFSP/FJSP 各 2–3 个公开项目检查检索和工程模式召回；
3. 建立“经典核心 / 常见变体 / 项目特有 / 实验上下文”人工标签；
4. 采集同实例受控候选，对四维权重和机制后验做校准；
5. 测量知识先验是否减少 Token、减少无关约束并提高项目增量召回；
6. 通过后才让 `use_for_optimization_attention` 影响 CIP、邻域和 Agent。
