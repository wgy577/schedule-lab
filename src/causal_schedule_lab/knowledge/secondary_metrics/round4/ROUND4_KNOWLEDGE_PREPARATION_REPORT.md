# Round 4 Knowledge Preparation Report

> 性质：知识层真实 ID 物化与规则整理  
> 状态：代码生成并通过结构校验；尚未接入运行时召回器，尚未做独立盲测  
> 数据来源：Round 3 的 55 个 canonical candidates、25 个 diagnostics、623 条 memberships、44 个 gaps

## 1. 本轮完成内容

本轮补齐了普通 GPT 因无法读取本地文件而未完成的工作。所有产物直接读取 Round 3 文件生成，未虚构 candidate ID，也未提升任何候选的证据状态。

- `gap_capability_groups.json`：把 44 个 gap 归入 9 个可重叠能力组。
- `membership_condition_recommendations.jsonl`：逐条覆盖全部 623 条 membership。
- `computability_requirements.jsonl`：为全部 55 个 canonical candidate 建立语义相关性与可计算性分离契约。
- `retrieval_reference_cases.jsonl`：保留 11 个 Round 3 架构验收场景，并补充 10 个组合、错误、缺字段、未知变体和角色定向场景，共 21 个。
- `round4_materialization_audit.json`：记录数量、引用完整性和状态不晋升检查。

## 2. 关键数量核对

| 检查项 | 结果 |
|---|---:|
| canonical candidate | 55 |
| diagnostic | 25 |
| membership 输入/输出 | 623 / 623 |
| 唯一条件 profile | 129 |
| coverage gap 已分组 | 44 / 44 |
| capability group | 9 |
| computability rows | 55 |
| reference cases | 21 |
| 全部自动检查通过 | true |

## 3. 关系条件的处理原则

每条 membership 被拆成四类条件：

1. `semantic_view_match`：判断候选是否与问题族、变体、机制、决策、角色或生命周期相关；
2. `computability_gate`：`required_ir_fields` 必须全部满足，才允许进入 computable channel；
3. `variant_applicability`：表达候选成立所需的变体；
4. `decision_controllability`：缺少可控决策时仍可保留语义候选，但标记为当前不可控。

重复激活条件通过 `condition_profile_id` 标识，后续程序可以抽成共享 profile。Round 4 不直接改写 Round 3 原始 membership。

## 4. 必须保留的限制

- 21 个参考场景是设计标签，不是独立真值。
- `currently_computable` 全部保持 `unknown_until_real_project_IR_is_checked`。
- `proposed` 不等于 `validated` 或 `causal`。
- diagnostic 目录与 55 个 canonical metric 目录保持身份隔离。
- 本轮没有实现真正的 deterministic retriever，也没有运行 Recall/Precision/NDCG 测试。
- 一个 gap 或候选允许出现在多个能力组，这是关系复用，不是实体复制。

## 5. 下一工程步骤

下一步应由 `causal_schedule_lab` 实现可执行召回器：读取项目画像，应用本轮条件，分别输出 semantic channel 与 computable channel，并通过独立 Critic/人工标签进行盲测。该步骤完成以前，不应继续宣称 Round 3 的 100% recall 是真实泛化结果。

## 6. 自动校验

详见 `round4_materialization_audit.json`。本报告生成时自动校验结果为：`all_checks_passed = true`。
