# 实验协议

> **所属模块**：H — 实验记忆与统计  
> **交叉分类**：D × E × F × H — 诊断候选的验证和统计验收  
> **文档职责**：定义配对实验、成本记录、统计检验与可复现要求。

## 1. 研究问题

实验的 P0 对象是甘特图/排程本身。代码问题只在它改变可达排程、求解正确性或 Oracle
可信度时作为安全门进入实验；论文和展示信息不作为在线优化奖励。

1. CIP 排序能否以更少 Full Oracle 调用发现有效改进点？
2. 责任路径和学习式闭包是否优于同规模随机区域？
3. Agentic 控制是否优于固定 VNS、固定 ALNS 与 solver-only？
4. 多保真后验是否在相同验证预算下取得更高改善？
5. 这些收益是否跨 JSP、FSP、FJSP、HFSP 保持，而非只适用于单一实例？

## 2. 数据划分

- 按实例而非候选划分 train/validation/test，防止同一 incumbent 泄漏。
- 四问题族分别分层；另设 leave-one-family-out 泛化实验。
- incumbent 来自多条确定性 dispatching rule 与外部稳定求解器。
- 每个 incumbent 记录 problem hash、schedule hash、rule、seed 和目标向量。

## 3. 对照方法

- incumbent，无改进；
- solver-only bounded neighborhood；
- deterministic VNS；
- bounded ALNS + tabu；
- 黑盒随机点 + 相同求解预算；
- CIP + fixed operator；
- CIP + Agentic control；
- 完整方法：语义 + CIP + 闭包 + Agent + multi-fidelity。

## 4. 因果对照

对每个真实 CIP 同时评估：

1. 同算子、随机位置；
2. 同位置、随机合法算子；
3. 同闭包规模、随机区域。

预算、solver seed、worker 数、deterministic time 和 Full Oracle 次数完全一致。

## 5. 消融

- 去项目语义；
- 去因果路径；
- 去闭包学习；
- 去 Agentic RL；
- 去多保真后验；
- 去因果对照训练；
- 仅 CP-SAT；
- GNN 换成 MLP/规则基线。

## 6. 指标

- 主目标相对改善与最终目标向量；
- feasible rate、acceptance rate；
- Full Oracle calls、time-to-first-improvement；
- improvement / wall-clock 与 improvement / Full Oracle；
- CIP Precision@k、Recall@k、NDCG@k；
- 闭包 precision/recall、outside-change rate；
- 风险分类 F1、有效性 Brier score、成本 MAE；
- 跨问题族泛化差距。

## 7. 统计

- 同实例同 seed 配对；
- 报告均值、中位数、95% bootstrap CI；
- 两方法使用 Wilcoxon signed-rank；
- 报告 Cliff's δ；
- 三种以上方法使用 Friedman，事后 p 值做 Holm 校正；
- 同时报告实际效应量，不只报告显著性。

## 8. 可复现性

- 固定 seed、单 worker；
- 使用 deterministic-time 或 conflict budget；
- 所有候选写入统一 JSONL；
- 保存软件版本、配置、problem/schedule hash；
- rejected candidate 也必须保存，避免幸存者偏差；
- 任何“已提升”结论必须能从原 incumbent 与记录重放。

## 9. LLM 项目语义盲测

- 案例必须来自网上公开论文与对应作者/官方代码仓库；
- 仓库固定到 commit，论文记录 URL 与 SHA-256；
- 论文只用于隐藏标签，代码侧请求禁止接触论文、标签和论文摘要；
- 标签生成、代码读取规划、代码分析使用彼此无状态的 API 请求；
- 代码文件必须先经 inventory，再从白名单选择；
- 逐字段报告 precision、recall、F1、exact，不只报告一个总分；
- LLM 生成的标签标记为 draft，只有人工审核后才能作为正式 ground truth；
- 同时报告 Token、延迟、模型版本、证据通过率和 Schema 修复次数；
- 失败案例和口径冲突必须保留，不能只展示识别正确的项目。
- 多模型比较必须共用固定标签；不得让每个模型生成自己的标签再给自己评分；
- API 强制温度、协议差异、中转身份边界和失败重试必须单独披露；
- 正式排名需要多案例、多次重复、失败率和方差；单案例单次只用于工程选型。
