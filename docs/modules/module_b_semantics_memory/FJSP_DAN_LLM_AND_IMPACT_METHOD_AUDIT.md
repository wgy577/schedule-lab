# FJSP-DAN LLM 调用与影响权重方法审计

> **所属模块**：B — 语义理解与知识记忆  
> **关联模块**：D — 诊断与因果机制  
> **交叉分类**：B × D — 语义结构化先验与影响因素诊断  
> **文档职责**：审计 LLM 调用链及其结构化先验，不冒充因果实验。

## Claude Code 实际做了什么

本轮一共 8 次模型调用，其中 Navigator 不是 Claude Code：

| 调用 | 模型 | 任务 | Token | 只读工具 |
|---|---|---|---:|---:|
| 1 | MiMo-v2.5-pro | 仓库导航 | 19,745 | 0 |
| 2 | Opus 5 | 项目/环境首轮 | 5,622 | 0 |
| 3 | Opus 5 | 项目/环境复核 | 5,301 | 0 |
| 4 | Opus 5 | 目标/约束 | 8,525 | 10 |
| 5 | Opus 5 | 决策/Oracle/未知项 | 10,575 | 11 |
| 6 | Opus 5 | 决策批次复核 | 8,206 | 0 |
| 7 | Opus 5 | 跨批次最终综合 | 9,518 | 0 |
| 8 | Opus 5 | 约束影响 Critic | 9,165 | 5 |

合计：

- 76,657 Token；
- 其中 MiMo 19,745，Opus 56,912；
- Claude Code 执行 26 次 `Read/Glob/Grep`；
- 0 次 JSON Schema 修复；
- Opus 报告成本约 4.84 USD。

### Token 高的原因

1. 所有 Opus 调用都使用 `reasoning_effort=max`，事件里的 output token 包含大量
   推理消耗，不等于最终可见 JSON 长度。
2. 三批分析不是三次调用，而是 5 次：两个批次进入了第二轮复核。
3. 最终综合再次读取证据包、短期记忆、导航、知识先验和完整 Schema。
4. 影响 Critic 又读取一次完整语义、知识先验和完整 Schema，并再次使用代码工具。
5. Claude Code 自带 `Read/Glob/Grep`，同时 Harness 还有自己的受控复读门，形成了
   两套读取机制。二者目的不同，但存在重复定位和重复上下文成本。
6. Navigator 输入了较完整的仓库清单和摘录，单次已消耗 19,745 Token。

这不是模型循环或失败重试；事件日志显示 8 次调用都正常结束。但当前链为质量优先
版本，成本明显高于必要下限。

### 后续降耗方向

- Navigator 先由程序 AST/依赖图压缩，便宜模型只为不确定文件定角色；
- 无补充证据时禁止自动进入第二轮；
- Claude Code 工具读取与 Harness ReadGate 二选一作为事实读取入口，另一层只做
  审批/审计；
- 三批输出通过确定性合并，只有冲突字段再调用最终 adjudicator；
- 影响权重先跑规则门，只把模糊项交给 Opus；
- 将知识树按命中祖先链裁剪，不再传完整 family/variant 内容；
- `max` 推理只用于约束、冲突和影响批次，导航与无冲突综合降低推理档位。

## 当前影响因子是怎么总结的

当前方法不是统计估计，而是“结构化 LLM 专家先验”：

```text
代码语义提取
→ 目标、决策、约束、环境
→ 检索问题族/变体知识祖先链
→ Opus 从固定选项中选择
   role
   context_scope
   feasibility_criticality
   decision_leverage
   objective_sensitivity
   candidate_discrimination
   controllability
   candidate variation
   optimization/diagnosis attention
→ 程序硬校验
→ 固定档位映射为 0/0.1/0.25/0.5/0.75/1
```

程序目前保证：

- 每个输入约束恰好出现一次；
- 不允许新造 constraint ID；
- 枚举之外的答案不能进入；
- `hard=true` 必须保留 `validation=required`；
- 输出仍为 `pending_human_review`；
- 报告明确标记 `assessment_kind=structured_llm_prior`、
  `empirically_validated=false`。

## 方法是否正确

作为“第一次筛选注意力”的方向正确，但不能叫最终影响因子，也不能叫因果权重。

当前主要缺口：

1. 它没有实际改变候选调度并测量 makespan；
2. 没有反事实对照、重复实验、置信区间或后验；
3. 容易混淆“约束存在”与“约束活跃程度”；
4. 容易把硬约束的重要性误写成候选区分度；
5. 评分档位由 LLM 选择，程序只做合法性检查，没有从代码自动推导；
6. 当前评分对象主要是 constraint，但真正可优化的常常是：
   decision → constraint activity/slack → mechanism → objective。

例如 `no_overlap` 在所有可行候选中都必须满足。真正区分候选的不是
“是否满足 no-overlap”，而是机器分配、机器序列、资源空闲间隔和关键路径等待。

## 建议的正式两阶段方法

### 阶段 A：结构先验

LLM 只做选择题，得到：

- 固定结构还是可控决策；
- 是否随候选变化；
- 直接控制还是间接控制；
- 对应的 slack/activity/mechanism；
- 需要什么 Oracle。

该阶段只负责缩小搜索范围。

### 阶段 B：经验校准

```text
选择一个可控 decision/factor
→ 生成局部受控候选
→ 保持其他变量尽可能一致
→ Static/Light/Full Oracle 验证
→ 测量 factor delta、mechanism delta、objective delta
→ 重复实例与反事实对照
→ 贝叶斯更新有效概率和收益分布
→ 得到可用于排序的后验权重
```

只有完成阶段 B，才能把 `structured_llm_prior` 升级为
`empirically_calibrated_impact`。
