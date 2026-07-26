# FJSP 语义与次级指标全部测试汇总（截至 2026-07-26）

## 1. 比较边界

本报告把测试分成三组，只有第三组可以直接比较当前 Harness 的模型差异：

1. 完整项目语义链：包含导航、代码复读和最终综合；
2. 历史自由输出 Critic：题库与 Schema 不同，只能作演进参考；
3. 当前同证据重放：固定 `fjsp_opus5_staged_semantics.json`，共享 6 条约束、4 个
   决策、85 项目录和 15 项程序合格候选；旧 final 输入 12 项，新高召回版输入全部
   15 项。

所有“最好”均指当前工程链的高召回候选质量，不代表已经通过人工真值、真实候选
扰动、Oracle 回放或统计检验。

## 2. 所有 FJSP 相关运行

| 顺序 | 运行 | 模型 | 状态 | Token | 延迟 | 次级目标 / 诊断 | 主要意义 |
|---:|---|---|---|---:|---:|---|---|
| 1 | 完整分阶段语义链 | MIMO Navigator + Opus-5 Analyst/Critic | 成功 | 76,657 | 多调用合计 | 旧 Schema 未直接保存目标 | 提取 6 约束、4 决策、3 Oracle；证据引用 100% 通过 |
| 2 | DeepSeek 历史自由 Critic | DeepSeek-v4-pro | 成功 | 18,202 | 116.0s | 3 / 3 | 无目录 ID 白名单，保留作旧基线 |
| 3 | GLM 历史 Critic | ark-code-latest | 失败 | 未返回 | 587.2s | 无 | 传输失败 |
| 4 | Opus 历史自由 Critic 首次 | Opus-5 | 失败 | 未返回 | 97.4s | 无 | 中转/客户端错误 |
| 5 | Opus 历史自由 Critic 重试 | Opus-5 | 成功 | 14,791 | 208.7s | 6 / 4 | 高召回，但名称和公式由模型自由生成 |
| 6 | MIMO 目录重放 v1 | MIMO-v2.5-pro | 成功 | 16,509 | 85.5s | 2 / 1 | 发现否定摘要误激活 setup/blocking |
| 7 | MIMO 目录重放 v2 | MIMO-v2.5-pro | 成功 | 15,758 | 87.1s | 3 / 2 | 指标召回变体门修正，但知识先验仍有词法污染 |
| 8 | Opus 目录重放 v2 | Opus-5 | 失败 | 未返回 | 134.6s | 无 | Cloudflare 524 |
| 9 | Opus 最小探针 | Opus-5 medium | 成功 | 36 | 3.9s | 不适用 | 证明模型、Token、官方客户端和路由正确 |
| 10 | Opus medium v3 | Opus-5 | Provider 成功、Schema 拒绝 | 7,906 | 111.4s | 未落盘 | 暴露次级目标缺少 `confidence` 契约 |
| 11 | Opus medium v4 | Opus-5 | 失败 | 未返回 | 240.0s | 无 | 本地请求上限；反映中转长请求波动 |
| 12 | Opus low v5 | Opus-5 | Provider 成功、Schema 拒绝 | 6,138 | 98.5s | 未落盘 | 仅多出空 `statement_note` 字段 |
| 13 | MIMO strict-final | MIMO-v2.5-pro | 成功 | 11,623 | 78.5s | 3 / 1 | 12 项输入，精确优先提示 |
| 14 | Opus strict-final | Opus-5 low | 成功 | 17,106 | 66.1s | 5 / 4 | 12 项输入，精确优先提示 |
| 15 | MIMO high-recall 首次 | MIMO-v2.5-pro | Provider 成功、安全门拦截 | 12,713 | 97.7s | 未落盘 | hard constraint 被标为 not_required |
| 16 | **当前 MIMO high-recall** | MIMO-v2.5-pro | **成功** | **12,857** | 83.6s | **8 / 1** | 15 项全输入；3 medium、5 low |
| 17 | **当前 Opus high-recall** | Opus-5 low | **成功** | **6,590** | 92.4s | **15 / 5** | 15 项全输入；3 high、6 medium、6 low |
| 18 | MIMO open-world 初版 | MIMO-v2.5-pro | 成功 | 13,432 | 95.8s | 7 + 1 novel / 2 | 尚未拆 novel diagnostic |
| 19 | Opus open-world 初版 | Opus-5 low | 成功 | 8,164 | 118.0s | 15 + 2 novel / 5 + 4 novel | 暴露 metric/diagnostic 混类 |
| 20 | MIMO open-world 双通道 | MIMO-v2.5-pro | 成功 | 14,802 | 105.1s | 5 + 0 novel / 1 + 1 novel | 资格一致性正确进入诊断通道 |
| 21 | Opus open-world 双通道 | Opus-5 low | 成功 | 9,050 | 128.0s | 15 + 2 novel / 4 + 3 novel | 推动增加候选变化结构门 |
| 22 | **MIMO open-world qualified-final** | MIMO-v2.5-pro | **成功** | **13,343** | 71.2s | **2 + 1 novel / 1 + 1 novel** | metric/diagnostic 正确分流 |
| 23 | **Opus open-world qualified-final** | Opus-5 low | **成功** | **9,184** | 130.8s | **14 + 2 novel / 4 + 3 novel** | 当前正式开放世界 Artifact |

Provider 成功但没有形成 Artifact 的运行仍记为失败：模型确实返回了内容，但程序没有
让不符合当时契约或安全门的内容进入正式结果。当前未知扩展字段会隔离并留痕，不再
导致整次失败；伪造 ID、重复 ID、未知约束引用和已知字段类型错误仍严格拒绝。

## 3. 当前高召回版完全同条件比较

| 项目 | MIMO high-recall | Opus high-recall |
|---|---:|---:|
| 固定输入语义 | 相同 | 相同 |
| 目录规模 | 85 | 85 |
| 程序合格候选 | 15 | 15 |
| 进入 Prompt | 15 | 15 |
| 次级目标 | 8 | 15 |
| 置信分层 | 3 medium、5 low | 3 high、6 medium、6 low |
| 诊断项 | 1 | 5 |
| Schema 扩展 / 安全覆盖 | 0 / 0 | 0 / 0 |
| Token | 12,857 | 6,590 |
| 延迟 | 83.6s | 92.4s |

两者共同选择 8 项，包括：

1. `assignment_processing_penalty`：机器选择加工时长机会损失；
2. `maximum_machine_workload`：最大机器工作负荷；
3. `machine_workload_cv`：机器工作负荷变异系数；
4. `interoperation_wait_total`：工序间等待总量；
5. `bottleneck_workload_excess`：瓶颈负荷超额；
6. `machine_idle_time_total`：机器空闲总量；
7. `critical_resource_internal_idle_time`：关键资源内部空闲时间；
8. `resource_capacity_idle_rate`：资源容量空闲率。

Opus 另保留稀缺资格负荷、机器队列等待总量、资源利用率变异系数、分配集中度 HHI、
平均连续活跃时长、多资源同步等待总量和在制品占用面积 7 项，其中 6 项被标为 low。Opus 还识别 5 个
独立诊断；MIMO 只保留 1 个。

## 4. 是否变好

### 开放世界增量

题库内选择之外，当前链允许代码证据支持的题库外提案。新次级目标必须明确属于候选
排程或构造轨迹，并说明同一实例候选变化依据与干预把手；实现/资格/Oracle/Validator
风险进入新诊断通道。最终 Opus 提出：

- `non-delay 掩码剪除对数量累积`（medium）；
- `下界 shaping 增量总量`（low，需先检查望远镜求和是否与终态下界重复）；
- 资格集合跨表示一致性、CP-SAT 状态/时限、独立可行性校验器三个新诊断。

它们全部固定为 `proposed`，既不会因不在选择题中被忽略，也不会未经审核进入优化器。

### 明确变好的部分

- `--reasoning-effort` 现在真实传给 Claude Code `--effort`，不再是假配置；
- 变体和工程模式先验由结构化 constraint/environment/objective 激活，不再从
  “未见 setup/双资源”等否定句中误召回；
- family 先验只注入已命中的变体，不再把全部兄弟变体塞入 Prompt；
- Critic Prompt 从约 25.5k 字符降到约 22.3k 字符；
- 次级目标 `confidence` 成为固定三档软权重；
- 未知扩展字段被隔离并保留路径和值，不再因模型多给解释而废弃整次结果；
- `hard=true → validation=required` 由程序所有，并记录任何安全覆盖；
- 输入上限从 12 增到 20，当前 15 个合格 FJSP 候选全部进入高召回选择；
- 当前 MIMO Token 从 15,758 降到 11,623；
- 当前 Opus 在目录白名单下形成 15 项分层候选池。

### 没有被证明的部分

- 15 个候选中哪些真正能预测或促进 makespan 改善；
- 指标排序的 Precision/Recall/NDCG；
- 哪个指标是真实影响因素；
- non-delay 解码限制应作为次级目标还是算法诊断；
- 当前结果能否跨其他 FJSP、JSP、FSP、HFSP 项目泛化。

## 5. 当前最好结论

`fjsp_opus5_open_world_qualified_final.json` 是当前最完整的 FJSP 入口 Artifact：它同时
保存14个目录内候选、2个题库外次级目标和7个目录内/外诊断，并把题库外内容限制在
代码证据、候选变化、干预把手和 proposed 状态之内。

因此它可以称为“当前最佳开放世界高召回入口 Artifact”，但不能称为“有效因素”或“已
验证最优指标集合”。下一步应优先计算 high 和 medium 项，同时以较低预算抽查 low
项；再通过同实例候选变化、受控干预和 Oracle 配对实验更新真实效应后验。

## 6. 当前正式产物

- `fjsp_opus5_metric_catalog_replay_final.json`；
- `fjsp_opus5_metric_catalog_replay_final.events.jsonl`；
- `fjsp_mimo_metric_catalog_replay_final.json`；
- `fjsp_mimo_metric_catalog_replay_final.events.jsonl`；
- `fjsp_opus5_debug_probe.events.jsonl`；
- `fjsp_mimo_metric_catalog_high_recall.json`；
- `fjsp_mimo_metric_catalog_high_recall.json.events.jsonl`；
- `fjsp_opus5_metric_catalog_high_recall.json`；
- `fjsp_opus5_metric_catalog_high_recall.json.events.jsonl`；
- `fjsp_mimo_open_world_qualified_final.json`；
- `fjsp_mimo_open_world_qualified_final.json.events.jsonl`；
- `fjsp_opus5_open_world_qualified_final.json`；
- `fjsp_opus5_open_world_qualified_final.json.events.jsonl`；
- 本报告。
