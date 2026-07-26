# Critic 模型成本、稳定性与 CC 冗余审计

> **所属模块**：H — 实验记忆与统计  
> **关联模块**：B — 语义理解与知识记忆  
> **交叉分类**：B × H — LLM 运行、Token、成本与失败审计  
> **测试日期**：2026-07-26  
> **审计边界**：只报告冗余，不在本轮修改 CC 分批、复读或 checkpoint 结构。

> **后续决策（2026-07-26）**：本文“DeepSeek 作为默认低成本 Critic”的结论已被
> 高召回优先策略取代。当前默认审核为 Opus；影响审核 CLI 暂时拒绝 DeepSeek，
> 不得由其删除、降权或最终否决候选指标。历史实测数据继续保留。

## 1. 测试条件

- 项目：已人工核验的 FJSP-DAN 代码快照；
- 强语义模型：Claude Opus 5；
- 候选 Critic：Opus 5、火山 GLM-5.2、DeepSeek-v4-pro；
- Critic 输入：同一份已核验 `SemanticAnalysis` 和问题族知识；
- 新约束：LLM 只选择指标 kind；指标名称、计算定义和单位由程序目录固定；
- 所有在线结果保留独立 JSONL 事件日志。

## 2. 实测结果

| 运行 | 结果 | 成功日志 Token | Schema repair Token | 延迟 | 已知费用 |
|---|---:|---:|---:|---:|---:|
| 完整链 + slash skill | 失败：CC 不识别 `/scheduling-code-semantics` | 19,100（仅 Navigator） | 0 | 54.4 s | Opus 0 |
| 完整链，关闭 slash skill | 失败：第三批修复后仍未通过 | 87,835 | 20,757 | 873.9 s | Opus $2.784，Navigator 未知 |
| Opus Critic 首次 | 失败：中转流中断 | 日志未形成完整 Token | 0 | 请求失败 | $0.099 |
| Opus Critic 重试 | 成功 | 14,791 | 0 | 208.7 s | $0.469 |
| GLM-5.2 Critic | 失败：3 次远端断开 | 无成功 Token | 0 | 587.2 s | 网关未返回 |
| DeepSeek-v4-pro Critic | 成功 | 18,202 | 0 | 116.0 s | 网关未返回 |

说明：Claude Code 日志把缓存创建 Token 与普通 input Token 分开，成功行只记录
`input_tokens + output_tokens`，因此 Opus 的 14,791 不能与 DeepSeek 的 18,202 做
完全同口径的输入成本比较。费用字段是更可靠的 Opus 成本参考。

## 3. 输出质量

Opus Critic 选择了 6 个次级指标和 4 个诊断项，覆盖更广，但曾自行改变“关键资源
内部空闲时间”的边界定义。这个问题已通过程序端指标目录消除。

DeepSeek Critic 一次通过 Schema，选择了：

1. 机器选择加工时间增量；
2. 最大机器工作负荷；
3. 关键资源内部空闲时间；
4. 机器资格、Oracle 和下界三个诊断项。

DeepSeek 比 Opus 保守，漏选了工序就绪后等待时间、受限解码规则完工期差值和低柔性
工序机器负荷，但没有创造宽泛指标，所有名称、定义和单位均与程序目录完全一致。
因此适合作为默认低成本 Critic；Opus 适合作为人工触发的扩展复核。

## 4. CC 是否冗余

结论：**存在明显冗余，但不是所有多轮读取都能直接删除。**

已确认的冗余：

- Harness 模式仍注入 slash skill，但本次 CC 进程不识别该命令，造成一次完整
  Navigator 重跑后才暴露错误；
- 关闭 skill 后的完整链产生 20,757 Schema repair Token，占该次总 Token 的 23.6%，
  占 Opus Token 的约 29.9%；
- 运行失败没有阶段 checkpoint，重试时 Navigator 再次读取相同仓库，两次 Navigator
  共使用 37,559 Token；
- Critic 已成为枚举选择任务，继续使用 Opus 生成长篇自由文本没有成本优势。

暂不能直接判为冗余的部分：

- “目标与约束”三轮可能包含必要的 AST 跨文件复读；当前日志没有记录每轮新增证据数，
  还不能区分必要补证与重复分析；
- 最终综合虽然重复阅读压缩事实，但承担跨批矛盾合并，不能在没有消融实验前删除。

下一轮若优化 CC Harness，应先增加每轮的新增事实数、新增证据数、重复事实数和
`Token / accepted new evidence`，再决定合并批次或提前停止。本轮未修改这些结构。

## 5. 当前模型策略

- 默认 Critic：DeepSeek-v4-pro；
- 可选 Critic：火山 GLM-5.2，但当前网关不稳定；
- 人工升级复核：Opus 5，仅用于低置信度、证据冲突或连续 Schema 失败；
- 禁止隐藏自动回退到 Opus，避免不可见成本。

## 6. 证据文件

- `outputs/fjsp_test/fjsp_deepseek_metric_critic.json`
- `outputs/fjsp_test/fjsp_deepseek_metric_critic.events.jsonl`
- `outputs/fjsp_test/fjsp_measurable_target_critic.json`
- `outputs/fjsp_test/fjsp_measurable_target_critic_retry.events.jsonl`
- `outputs/fjsp_test/fjsp_glm52_metric_critic.events.jsonl`
- `outputs/fjsp_test/fjsp_measurable_targets_no_skill.events.jsonl`
