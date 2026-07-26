# 项目约束图与调度知识继承树

> **所属模块**：B — 语义理解与知识记忆  
> **交叉分类**：B × D — 语义约束图、变体路由与诊断输入  
> **文档职责**：定义项目语义图、知识继承树及 Variant Head 路由。

## 两张图解决不同问题

### 1. 项目约束图（每个项目一张）

它不是传统析取图的替代命名，而是更大的证据图：

```text
Project
├── ProblemFamily
├── Environment
├── Objective
├── Constraint
├── Decision
├── Oracle
├── Unknown
└── CodeEvidence

IR Instance
├── Job ──CONTAINS──> Operation ──PRECEDES──> Operation
├── Operation ──HAS_MODE──> ProcessingMode ──REQUIRES──> Resource
├── Constraint ──APPLIES_TO──> Operation / Resource / Job / Mode
├── Objective
└── ChoiceBinding ──BINDS──> Operation
```

传统析取图主要表达工序先后与资源冲突；本图额外表达环境、候选机器/模式、显式约束、
决策自由度、目标、Oracle、未知项和逐条代码证据。

为避免审计信息淹没调度结构，同一数据现在分成多个视图：

- `scheduling_core`：问题族、工件、工序、模式、机器、加工顺序、约束、决策、
  主目标、次级目标与交叉关系；
- `context_overlay`：静态/动态、确定/随机、特殊运行条件；
- `validation_overlay`：求解器、Validator、Simulator、Oracle；
- `evidence_overlay`：代码文件、symbol 和引用；
- `audit_overlay`：未知项、矛盾和待人工确认内容。

默认面向 Agent 的图应先读取 `scheduling_core`，需要验证或追溯时再按需展开 overlay。
原始信息不删除，但不会全部混在主调度图里。

证据政策：

- LLM 结果标记为 `relation_basis=llm_finding`；
- IR 明确关系标记为 `relation_basis=ir`；
- 代码引用标记为 `relation_basis=explicit_citation`；
- 当前版本不自动创造无证据的因果边；
- 后续因果边必须来自规则、实验或人工审核，并保留 provenance。

实现：

- `semantic_graph_models.py`
- `semantic_graph.py`
- `LLMSemanticCompilation.constraint_graph`

### 2. 调度知识继承树（跨项目共享）

```text
车间调度（L1：共同实体与共同语义）
├── JSP（L2：只存 JSP 增量）
│   ├── classic JSP（L3：空变体增量）
│   ├── release-date / dynamic JSP
│   ├── setup JSP
│   ├── transport JSP
│   ├── no-wait / blocking JSP
│   ├── dual-resource JSP
│   └── stochastic / robust JSP
├── FSP
│   ├── classic FSP
│   ├── permutation FSP
│   ├── non-permutation FSP
│   ├── no-wait / blocking FSP
│   ├── setup FSP
│   ├── distributed FSP
│   └── reentrant FSP
├── HFSP
│   ├── classic HFSP
│   ├── unrelated / eligibility HFSP
│   ├── blocking / buffer HFSP
│   ├── setup HFSP
│   ├── no-wait HFSP
│   ├── batch HFSP
│   └── transport HFSP
└── FJSP
    ├── classic FJSP
    ├── partial / total flexibility
    ├── dynamic FJSP
    ├── distributed FJSP
    ├── transport FJSP
    ├── dual-resource FJSP
    ├── setup / maintenance FJSP
    ├── energy / multi-objective FJSP
    └── stochastic / robust FJSP
```

SQL 中每个节点只保存 `local_delta`，使用 `inherits_from` 指向唯一父节点。查询时按
根到叶顺序合并，并同时返回 `lineage` 和 `contributions`，因此可以知道每项知识来自
哪一级，而不是得到一份无法追踪来源的扁平 JSON。

```bash
causal-schedule-lab memory-build

causal-schedule-lab memory-taxonomy \
  --node "partial flexibility" \
  --family FJSP
```

禁止建立“各类变体”“其他变体”等概括节点。每个新类型必须建立独立、可命名、
可引用证据、只保存本层增量的分支。组合变体或工程子类型可以继续成为第四级、
第五级或更深层节点，数据库结构不需要改变。

实现：

- `taxonomy.py`
- `storage/graph_store.py`
- `storage/memory.py`
- `knowledge/scheduling_families.json`

## 与语义 Agent 的关系

1. Navigator 和分批 Analyst 读取代码并产生有证据的语义发现；
2. FSP/JSP/HFSP/FJSP 分类用于命中知识树；
3. 祖先合并只提供经典知识先验，不作为当前项目的代码证据；
4. 最终综合生成项目约束图；
5. 人工可沿 CodeEvidence 回查每个项目事实；
6. 后续优化器只使用审核通过、可控制且能被 Oracle 验证的部分。

继承树在 Agent 中的作用不是直接给答案，而是：

```text
识别到 FJSP
→ Variant Head Router 只扫描少量高显著度标签
→ 命中一个具体 L3；无命中则选择 classic FJSP
→ SQL 只解析 车间调度→FJSP→所选 L3 的 lineage
→ 用该分支生成少量代码验证选择题
→ 只把代码证实的内容写入项目图
```

不会把 FJSP 的全部变体兄弟节点同时交给 Agent。这样既避免上下文膨胀，也避免模型
为了回应候选列表而在代码里“寻找”并不存在的约束。Head 是路由标签，不是项目事实；
命中后仍需代码证据确认。Head 定义位于 `knowledge/variant_heads.json`，新增具体变体时
必须同时增加少量、具有区分度的 Head，不能使用“特殊情况”等宽泛标签。

从 0.6.2 起，`LongTermMemoryContext.taxonomy_profiles` 只把所选 L3 的 lineage、
resolved 和 contributions 注入分批 Agent Prompt。知识树仍是先验，不能替代代码证据。
