# LLM 项目语义盲测 Harness

> 当前阶段：第一版公开论文—官方代码盲测链已运行  
> 主线：强模型 API + 可审计 Harness  
> 训练路线：保留在 `CODE_ONLY_SEMANTIC_LEARNING.md`，当前不作为主线

## 1. 目标

测试强 LLM 在**看不到论文**时，能否仅根据完整或碎片化代码识别调度项目的：

- 项目类型与问题族；
- 环境假设；
- 优化目标；
- 硬约束；
- 决策变量与允许干预；
- 可用 Oracle。

对于 JSP/FSP/HFSP/FJSP，代码分析端现在可以使用公开的经典问题族知识先验，但
知识库不含当前案例论文结论，且不能作为项目 evidence。评估将逐步从“是否重复列出
所有经典字段”转向“是否准确识别经典问题族以及项目新增变体和优化杠杆”。

论文只用于生成隐藏标签，不能进入代码侧上下文。标签和预测都先保持
`pending_human_review`；LLM 自己生成的论文标签不能冒充人工真值。

## 2. 第一例网上公开案例

- 论文：[NeurIPS 2020, *Learning to Dispatch for Job Shop Scheduling via Deep
  Reinforcement Learning*](https://proceedings.neurips.cc/paper_files/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf)；
- 代码：[作者公开的 L2D 官方 PyTorch 仓库](https://github.com/zcaicaros/L2D)；
- 代码版本：`7b2efbb1ffc960260b16952f2bed68e500765bf0`；
- 案例清单：`examples/harness_cases/l2d.json`；
- 下载内容：`external_cases/l2d/`，被 `.gitignore` 排除，不复制进本项目源码。

仓库中的 `paper/` 目录在 inventory、读取规划和最终代码证据包三处都被排除。

## 3. 隔离流水线

```text
论文 PDF
  → 独立无状态 API 请求
  → PaperSemanticLabel（draft）
  → 引用必须指向 paper.pdf

官方代码固定 commit
  → 程序建立 inventory
  → 独立无状态 API 请求选择最多 24 个文件
  → 白名单证据包（再次排除 paper/）
  → 独立无状态 API 请求
  → Code SemanticAnalysis
  → 程序核验文件与 symbol

Paper label + Code prediction
  → 本地集合评分
  → 每字段 precision / recall / F1 / exact
  → 人工复核差异
```

三个 API 请求不共享消息历史。代码侧 Prompt 不包含论文文本、论文摘要或论文标签。
测试会直接检查论文文本和仓库内隐藏论文内容没有出现在规划与预测请求中。

## 4. 稳定控制点

| 控制点 | 实现 |
|---|---|
| 输入来源 | URL、固定 commit、文件 SHA-256 写入案例和产物 |
| 文件选择 | 先程序 inventory，再让 LLM 从白名单做选择题 |
| 分类空间 | 固定枚举；不允许模型创造问题族、约束或决策类型 |
| 输出 | Pydantic JSON Schema，禁止额外字段 |
| 推理质量 | 所有语义请求默认开启模型支持的最高推理档；具体协议见 `LLM_SEMANTIC_COMPILER.md` |
| 随机性 | 非推理模型使用 `temperature=0`；推理模型不发送可能冲突的 temperature |
| 格式错误 | 最多一次定向 JSON/选项修复 |
| 论文泄漏 | `excluded_code_paths` + 选择白名单 + 请求内容测试 |
| 证据 | 每项引用真实文件和 symbol；程序核验 |
| 成本 | 标签、规划、预测分别记录模型、Token、延迟和调用次数 |
| 结论 | draft 标签与预测均需人工审核，不自动进入正式语义 DSL |

## 5. 评分定义

第一版对八组离散语义做集合比较：

1. `project_type`；
2. `problem_families`；
3. `environments`；
4. `objectives = kind:sense`；
5. `constraints = kind:scope:hard`；
6. `decisions = kind:modifiable`；
7. `oracles = kind:required`；
8. `allowed_interventions`。

报告每组 precision、recall、F1 和 exact match，再计算 Macro-F1 与 exact-match
rate。这个指标衡量枚举语义一致性，不衡量自由文本相似度。

## 6. 首次真实盲测

运行日期：`2026-07-24`。

| 项目 | 结果 |
|---|---:|
| 请求模型 / 实际模型 | `ark-code-latest` / `glm-5.2` |
| 论文标签 Token | 22,650 |
| 代码读取规划 Token | 2,477 |
| 代码识别 Token | 22,094 |
| 总 Token | 47,221 |
| 论文标签 / 规划 / 代码识别延迟 | 116.596 / 17.868 / 191.610 秒 |
| 问题族、环境、目标 F1 | 1.000 / 1.000 / 1.000 |
| 约束、决策 F1 | 0.750 / 0.500 |
| 项目类型、Oracle F1 | 0.000 / 0.000 |
| Macro-F1 / exact-match rate | 0.615 / 0.375 |
| 论文标签引用 | 100% 通过 |
| 代码引用 | 30/30 通过 |

真实结果表明，模型对 JSSP、确定性离线环境和 makespan 目标识别稳定，但对
“调度优化项目还是 ML+调度混合项目”、隐式序列决策和 Oracle 边界存在明显口径
差异。这些差异正是后续 Harness 需要拆成更细选择题并交给人审核的部分。

产物：`outputs/l2d_semantic_harness.json`。

同一固定标签上的 GLM、DeepSeek、MiMo、Kimi 和 Opus 对比见
[L2D_MODEL_COMPARISON_REPORT.md](L2D_MODEL_COMPARISON_REPORT.md)。

## 7. 运行

第三方论文和代码由用户从原始来源下载到案例清单声明的忽略目录。准备完成后：

```bash
.venv/bin/causal-schedule-lab run-semantic-harness \
  --case examples/harness_cases/l2d.json \
  --env-file .env \
  --provider-prefix SEED \
  --max-paper-characters 36000 \
  --max-code-characters 42000 \
  --output outputs/l2d_semantic_harness.json
```

## 8. 下一步

- 先按
  [L2D_PAPER_CODE_HUMAN_VERIFICATION.md](L2D_PAPER_CODE_HUMAN_VERIFICATION.md)
  分别审核论文主张、代码事实和二者对齐关系；
- 人工审核完成前，保留 `draft_llm_label`，不把旧 F1 当成模型真实能力排名；
- 审核通过后再生成 `human_verified_label`；
- 将项目类型拆为“业务任务类型”和“实现技术类型”，避免 `mixed` 口径歧义；
- 明确 Oracle 是“环境转移”“可行性检查器”还是“外部权威验证器”；
- 将 `select`、诱导的 `sequence` 和环境派生的 `start_time` 分层；
- 为 JSP/JSSP、capacity-1/no-overlap 等建立受审核的语义等价归一化；
- 增加证据定位到行号与摘录的核验；
- 从网上加入 FJSP、FSP/HFSP、RCPSP 等论文—官方代码案例；
- 在多个模型和 Prompt 版本上运行同一固定案例集；
- 再决定是否需要训练小模型或做 Teacher–Student 蒸馏。
