# 无论文与碎片化文档项目的语义学习方案

> 目标：训练时利用论文与代码配对数据，部署时只依赖代码或少量碎片文本  
> 当前状态：架构与训练协议已定义，数据集和训练实现尚未开始  
> 原则：论文是训练阶段特权信息，不是生产环境必需输入

## 1. 目标场景

生产环境按信息完整度分为：

| 场景 | 可用输入 | 目标 |
|---|---|---|
| A. 完整配对 | 论文、代码、实例、结果 | 生成高质量教师标签与研究基线 |
| B. 代码加碎片 | 代码、README、注释、零散配置 | 主要部署场景 |
| C. 纯代码 | 代码、配置、测试 | 必须具备的最低能力 |
| D. 不完整代码 | 部分模块、接口或二进制依赖 | 输出可证实部分和明确 unknown |

系统不能因为缺少论文就拒绝工作，也不能用猜测填补缺失信息。

## 2. Teacher–Student 总体路线

在 Teacher–Student 之前先使用可审计的调度问题族知识检索。经典 JSP/FSP/HFSP/
FJSP 的定义、核心约束和常见变体不应依赖每个项目重新生成；训练重点应放在问题族
与变体检索、项目增量语义、约束的情境化优化杠杆，以及 unknown/冲突/领域 Oracle。

```text
论文 + 代码 + 实例
        │
        ▼
Teacher LLM 多轮语义编译
        │
        ├── PaperSemantics
        ├── CodeSemantics
        ├── Paper↔Code 对齐矩阵
        └── 人工裁决标签
                 │
                 ▼
       SemanticTrainingCase
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
完整视图训练         模态缺失/碎片化视图
       │                   │
       └─────────┬─────────┘
                 ▼
       Code-only Student
                 │
                 ▼
生产项目：代码 + 可选碎片文本
```

Teacher 可以使用高能力 API 模型。Student 可以是：

- 同一 LLM 的受约束 Agent 工作流；
- 可微调的开源 Code LLM；
- 代码图编码器与分类/检索头；
- LLM 与小型证据排序模型的混合系统。

第一阶段先建立 LLM Agent 基线，再决定是否进行 SFT/LoRA 或训练专用编码器。

## 3. 训练数据单元

每个 `SemanticTrainingCase` 至少包含：

```text
case_id
paper_id / paper_hash
repository_url / commit
license
problem_family
option_library_version
paper_semantics
code_semantics
alignment_matrix
adjudicated_semantics
evidence_citations
code_only_view
sparse_text_views
unknown_labels
split
```

标签必须追溯到论文段落、代码 symbol、配置、测试、实例或人工裁决。论文不是默认
真值：每个配对案例必须分别保存 `PaperClaims` 与 `ImplementationTruth`，并用
`paper_and_code / paper_only / code_only / representation_equivalent /
conflict / ambiguous` 描述对齐。未经人工确认的论文标签不得作为 Student 的正式
监督标签。

## 4. 自动生成缺失模态训练视图

从完整配对案例生成多个退化视图：

1. 移除论文，只保留完整代码；
2. 保留代码和 README，移除论文；
3. 只保留一部分 README/注释；
4. 随机遮蔽配置文件；
5. 随机遮蔽测试；
6. 随机移除高价值模块；
7. 保留接口，移除实现；
8. 引入论文与代码轻微冲突样本；
9. 加入无关文档和噪声文件；
10. 对关键事实设置 `unknown`。

同一个案例的所有视图必须留在同一个数据划分，禁止跨 train/test 泄漏。

## 5. Student 需要学习的任务

### 5.1 项目级选择题

- 项目类型；
- 问题族；
- 环境类型；
- 目标类型和方向；
- 约束类型和范围；
- 决策类型；
- Oracle 类型；
- 是否需要人工审核。

### 5.2 证据检索

给定语义问题，从仓库中选择：

- 相关文件；
- 相关 symbol；
- 相关测试；
- 相关配置；
- 最小证据集合。

### 5.3 结构化解释

只在已选选项之后生成：

- 具体含义；
- 作用范围；
- 证据说明；
- 尚缺什么；
- 建议向用户提出的问题。

### 5.4 Unknown 与冲突识别

Student 必须学习：

- 证据不足时选择 `unknown`；
- 文件之间矛盾时选择 `conflict`；
- 不把注释当成执行事实；
- 不把计划文档当成已运行实现；
- 不把函数名当作约束已经生效的证明。

## 6. 模型与损失

建议组合：

```text
Repository Retriever
  + Code/Graph Encoder
  + Constrained Choice Heads
  + Evidence Ranker
  + LLM Explanation Decoder
  + Confidence Calibrator
```

总损失可以写为：

\[
\mathcal L =
\lambda_c\mathcal L_{choice}
+\lambda_e\mathcal L_{evidence}
+\lambda_d\mathcal L_{distill}
+\lambda_u\mathcal L_{unknown}
+\lambda_x\mathcal L_{consistency}
+\lambda_r\mathcal L_{retrieval}
\]

其中：

- \(\mathcal L_{choice}\)：选项分类交叉熵；
- \(\mathcal L_{evidence}\)：证据文件/symbol 排名损失；
- \(\mathcal L_{distill}\)：Student 对齐完整信息 Teacher 分布；
- \(\mathcal L_{unknown}\)：证据不足校准；
- \(\mathcal L_{consistency}\)：完整视图和退化视图语义一致性；
- \(\mathcal L_{retrieval}\)：正确代码片段对比学习或 pairwise ranking。

论文只进入 Teacher 和标签生成，不进入 code-only Student 的推理输入。

## 7. 可训练的能力分层

### L0：无需训练的强基线

- AST、依赖、配置和测试索引；
- GLM-5.2 多轮工具检索；
- 选项库与严格 JSON；
- 证据审计；
- 人工审核。

### L1：训练检索与排序

- 哪些文件最值得先读；
- 某个约束对应哪些 symbol；
- 哪些测试能验证某个声明；
- 在 Token 预算内选择最小证据集。

这是最先值得训练的部分，成本低且能明显改善大型仓库理解。

### L2：训练语义选择头

- 问题族、目标、约束、环境和决策分类；
- unknown/conflict；
- 置信度校准。

### L3：蒸馏或微调 Code LLM

- 使用 Teacher 生成并经人工裁决的轨迹做 SFT/LoRA；
- 使用拒绝样本训练证据忠实度；
- 可选偏好优化，奖励正确选项、有效证据和主动承认 unknown。

### L4：训练 Agent 检索策略

状态是当前证据覆盖，动作是“下一步读取哪个文件、运行哪个测试、询问什么问题”。
奖励同时考虑：

- 语义正确率；
- 证据覆盖；
- unknown 校准；
- Token 和时间成本；
- 无效工具调用；
- 人工审核负担。

## 8. 评估协议

必须报告三种输入条件：

```text
Full:       论文 + 代码
Sparse:     代码 + 碎片文本
Code-only:  只有代码/配置/测试
```

主要指标：

- 选项 macro/micro F1；
- 硬约束 precision/recall；
- 证据检索 Recall@K、MRR；
- 引用有效率；
- unknown AUROC/F1；
- conflict 检出率；
- 置信度 ECE/Brier；
- Token、延迟和费用；
- 人工修正数量；
- 最终 adapter/validator 构建成功率。

必须按仓库划分训练、验证和测试，并增加：

- leave-one-problem-family-out；
- leave-one-codebase-style-out；
- paper-hidden evaluation；
- 文档缺失强度曲线；
- 噪声与冲突鲁棒性。

## 9. 防止论文信息泄漏

Code-only 测试不只是删除 PDF，还要检查：

- README 是否大段复制论文定义；
- 配置名是否直接泄漏问题族；
- 文件名是否包含标签；
- 论文作者提供的结果文件是否暴露答案；
- 同一仓库 fork 是否跨数据划分；
- 同一论文不同版本是否跨数据划分。

必须保存 `leakage_manifest`，记录每个视图删除或保留了什么。

## 10. 用户与 Codex 分工

### 用户

- 收集论文、对应仓库、实例和许可证；
- 对 Teacher 标签和冲突项做抽样审核；
- 提供训练算力或允许使用的 API 预算；
- 决定公开、论文和商业使用边界。

### Codex

- 建立案例 manifest 和解析器；
- 生成 Paper/Code/Alignment 标签；
- 生成退化视图并执行防泄漏划分；
- 实现检索、分类、蒸馏和评估；
- 记录数据 hash、配置、checkpoint 和指标；
- 将通过验收的 Student 接入 Agent Provider。

## 11. 验收门

在声称“无论文也能理解项目”前，至少满足：

1. Code-only 测试集来自完全未见仓库；
2. 不使用论文、论文复制文本或结果标签；
3. 硬约束优先保证 precision，避免凭空增加约束；
4. unknown 和 conflict 有独立评价；
5. 引用必须指向真实代码、配置或测试；
6. 相比 GLM-5.2 单轮基线有统计显著改善，或在相同质量下降低成本；
7. 所有结论按问题族分别报告；
8. 输出仍需经过程序验证和人工审核。

## 12. 下一步实施顺序

1. 定义 `CaseManifest`、`PaperSemantics`、`CodeSemantics`；
2. 定义 Alignment 和 adjudication JSON Schema；
3. 支持 PDF/补充材料解析；
4. 支持大型仓库多轮检索；
5. 建立首批 10–20 个论文+代码配对案例；
6. 生成 code-only 与 sparse 退化视图；
7. 训练文件/symbol 检索器；
8. 建立 GLM-5.2 code-only Agent 基线；
9. 再决定训练分类器、GNN 或微调 Code LLM；
10. 执行独立仓库测试与消融。
