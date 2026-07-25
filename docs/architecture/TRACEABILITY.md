# 框架 1–24 节追踪矩阵

该表以《项目条件化因果核心点发现与 Agentic 调度改进框架（最终版）》为验收
基准。状态区分“代码已实现”和“需要真实数据才能形成论文结论”，避免把接口存在
误写成实验已经完成。

| 章节 | 工程状态 | 代码/协议落点 |
|---|---|---|
| 1–3 问题、目标、原则 | 已实现 | `controller.py`：从 incumbent 局部改进、失败回退、历史最好解单调 |
| 4 共享因果骨架 | 已实现 | `graph.py`：A→L→W→T→D→J 特征和异构关系 |
| 5 项目语义适配 | 部分实现 | 程序索引、证据门和火山 LLM 单轮编译已运行；多轮理解和人工审核尚缺 |
| 5.1 LLM 项目语义编译器 | 单轮主链已实测 | GLM-5.2 输出受约束 JSON；程序核验证据，结果保持待人工审核 |
| 5.2 统一 IR | 已实现 | `ir.py`、`io.py`；无外部工程依赖 |
| 5.3 长期知识记忆 | 独立代码已实现 | `storage/graph_store.py`、`storage/memory.py`：SQLite 图谱、L0–L4 FTS5、证据生命周期与导入；默认语义链尚未接入 |
| 6 CIP 四元定义 | 已实现 | `models.py`：D、R、P、Ω |
| 7 联合框架 | 已实现 | `controller.py`：CIP→Agent→repair→Oracle→accept/revert |
| 8.1 候选召回 | 已实现 | `cip.py`：关键资源空档、高等待、阻塞与族特征召回 |
| 8.1a 机制目标与资格门 | 独立代码已实现 | `mechanisms.py`：八类测量、候选变化门、直接/验证间接控制和保守后验；真实效应待采集 |
| 8.2 图/MLP 编码 | 已实现 | `learning.py.RelationalEncoder`；规则排序为无数据 fallback |
| 8.3 多任务头 | 已实现 | gain、validity、cost、risk、rank、closure、path |
| 8.4 反事实监督 | 已实现 | `counterfactuals.py`、`dataset.py`、pairwise ranking |
| 8.5 闭包预测 | 已实现 | 规则 1–3 级闭包 + 节点 membership head + sparse loss |
| 9 Agentic RL | 已实现 | `agentic_rl.py` 完整三元动作；`agent.py` BC/PPO |
| 9.1 短时程 | 已实现协议 | retry/expand/next/backtrack/full/stop，训练 horizon 由配置限制 3–10 |
| 9.2 Mask | 已实现 | 语义推荐、闭包等级、预算与失败历史联合 Mask |
| 10 条件生成 | 已实现 | `conditional_generator.py` 部分排程训练、multi-candidate、solver fallback |
| 10.1 闭包外冻结 | 已实现并测试 | `solvers/cp_sat.py` 精确固定 mode/start/end |
| 11 多保真验证 | 已实现 | `validation.py` Code/Static/Light/Full |
| 11.1 多保真后验 | 已实现 | `posterior.py` 有效性 Beta 后验、proxy bias、gain/cost acquisition |
| 11.2 因果对照 | 已实现 | `counterfactuals.CAUSAL_CONTROLS`、`experiments.py` |
| 12 在线闭环 | 已实现 | `AgenticImprovementController.run` |
| 13 分阶段训练 0–8 | 流水线已实现 | 语义→反事实→排序→路径/闭包→生成器→BC→offline RL→PPO→LLM 反思蒸馏 |
| 14 数据与日志 | 已实现 | `ExperimentRecord`、`audit.py`、`dataset.py` |
| 15 领域案例 | 通用化实现 | 不硬编码原文案例；通过 manifest/adapter/oracle 重现同一接入模式 |
| 16 实验设计 | 已实现协议 | `experiment_runner.py`、`docs/experiments/EXPERIMENT_PROTOCOL.md` |
| 16.1 统计检验 | 已实现 | bootstrap、Wilcoxon、Cliff's δ、Friedman、Holm |
| 16.2 消融 | 已实现矩阵 | `experiment_runner.ABLATIONS` |
| 17 创新点 | 架构已落地 | 项目条件语义、CIP、闭包、短时程控制、多保真证据 |
| 18 相关工作边界 | 已记录 | README 与实验协议不把框架等同于黑盒 LNS/LGS/NDS |
| 19 LGS | 可选扩展 | 保留多候选生成接口；按原文要求不作为首要依赖 |
| 20 理论边界 | 已实现 | 硬约束、严格改善、回退、有限预算；不宣称全局最优 |
| 21 第一版模块 | 已覆盖且超出 MVP | 语义、IR、CIP、闭包、Agent、repair、Oracle、logging、experiments |
| 22 论文贡献 | 待真实实验 | 代码允许验证；结论不能在跑完基准前预写 |
| 23 风险应对 | 已实现 | 证据门、闭包扩张、失败标签、solver fallback、多保真成本控制 |
| 24 参考工作 | 设计对齐 | 只吸收思想，不声称复现外部论文模型 |

## 自动验收层

| 层 | 验收内容 |
|---|---|
| 单元测试 | IR 引用、四问题族、确定性 dispatcher、统计与模型头 |
| 约束测试 | completeness、precedence、eligibility、duration、capacity、choice link |
| 局部修复测试 | 闭包外 mode/start/end 完全不变 |
| 学习 smoke | 多任务头前向、loss、反向传播 |
| 系统 smoke | 四问题族从 manifest 到 CIP、Agent、repair、Full gate |
| 研究验收 | 多实例、多 seed、基线/消融/统计；由实验产物而非 README 宣称 |
