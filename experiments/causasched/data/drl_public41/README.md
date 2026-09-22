# DANIEL 生成的公开 FJSP 初始解 bank（供 T2-M 训练使用）

打包日期：2026-09-18。来源运行：AutoDL
`/root/t2m_paper_tables_probe10`，RUN_ROOT =
`outputs/paper_public_fjsp_probe10_20260917_132822`，seed=9162026。

## 内容

| 目录 | 说明 |
|---|---|
| `daniel_sample100/` | 41 例公开 FJSP 实例，DANIEL 采样 100 个调度取最优；每个实例一个 `<id>.json`（problem + 最终 schedule + 全部 100 个采样 makespan），一个 `<id>.samples.json.gz`（100 个采样调度全文） |
| `daniel_greedy/` | 同 41 例的 DANIEL 贪心解码结果（作为较弱起点的备选） |

两个目录各含 `protocol.json`：bank 清单（41 条 entry，逐文件 SHA-256，
`complete: true`）。`paper_evaluate_drl_bank.py` 即按此协议加载并校验。

## 实例与来源

- 41 例 = Brandimarte Mk6/8/9/10/14/15 + Hurink Rdata19--38 + Behnke--Geiger
  1--15，与 `public_fjsp_manifest_20260917.json` 的 11 组划分一致。
- DANIEL 权重：`paper_baselines/daniel/trained_network/SD2/20x10+mix.pth`
  （SHA-256 见 bank 的 `protocol.json` 中 `weights_sha256` /
  `source_sha256`），官方预训练权重直接外推，未做微调。
- 全部 schedule 已通过 `causal_schedule_lab.core_validation.validate_schedule`
  可行性校验（评测脚本对 bank 与每轮搜索解均强制校验）。

## 已核对的数值事实

- `daniel_sample100` 的逐实例 makespan 与服务器 round 1 的 CausaSched
  起点完全一致（共享初解）。
- 41 例起点均值约 666.2；同批 CP-SAT（60 s/例）参考均值见
  probe10 结果包。
- 服务器 round 1（默认算子池，8 branches x 10 步）3158 步实际执行、
  仅 1/41 实例改进——即该策略在此类强起点上几乎无增益，这正是本次
  重新训练的动机。

## 建议的训练用法

1. 将 41 个 `problem + schedule` 对作为初始状态来源（等价于训练侧的
   S0/bank 注入），让策略从强局部最优附近学习改进，而不是只从弱起点学。
2. 100 个采样调度（`*.samples.json.gz`）可用作每实例的状态池多样性来源
   或课程式起点（弱到强）。
3. `daniel_greedy` 可作为对照组起点，验证"起点强度 vs 改进空间"的关系。
4. 上传后建议先跑 `sha256sum -c` 式核对（bank protocol 内含逐文件哈希），
   防止传输损坏。
