# CausaSched — V13

128实例、200步固定回合的调度改进训练与推理。 Scheduling improvement with fixed 128-instance training episodes.

![Problem](https://img.shields.io/badge/Problem-Scheduling-17324D?style=flat-square) ![Method](https://img.shields.io/badge/Method-Causal_RL-2F6B5F?style=flat-square) ![Objective](https://img.shields.io/badge/Objective-Makespan-C47C3C?style=flat-square)

## 当前V13配置

- `data/train128/protocol.json`固定128个TRAIN实例：原26个全部保留，另102个来自旧200实例训练检查点。
  按类型轮询、ID哈希排序抽取，不按改善量挑选；相同结构去重，文件附SHA256。
  FJSP 64、JSP 22、FSP 21、HFSP 21；类型沿用原数据标注。
- 每次更新全部128图×16条轨迹；16 worker分批处理，GPU微批次16。
- 每张图每个大回合200步分配预算，统一10步或20步小段；全部完成后同时回到各自固定初始规则解。
  早停仍消费分配预算，不承诺每条轨迹执行满200个动作。网络与优化器连续学习。
- 默认只输出启动信息、阶段总结、每cycle一条summary及保存路径；详细父进程日志在`detail.log`。
  `rollout_metrics.jsonl`、`reward_components.jsonl`、audit和TensorBoard保留。`--verbose`恢复详细终端输出。
- summary含stop原因统计。仅增加可观测性，未修改重复状态或执行失败时的早停行为。
- L=6、单+双算子、负载奖励幅值目标35%、工期退化惩罚0.2保持。
- 这128个实例已经用于训练，不能再作为未见测试集；特别是原26个公开实例不能用来声称未见泛化。

## 安装及启动

```bash
pip install -r requirements_e2e.txt
# 新训练；需要先在 inference_assets/runtime.pt 放好私有基础运行时
OUTPUT=outputs/train128 bash run_e2e_single.sh

# 从26图检查点升级：保留网络/优化器/已有全局最好解，明确开启新回合
OUTPUT=outputs/train128_resume bash run_e2e_single.sh \
  --resume /path/to/latest.pt --expand-training-cohort \
  --reset-episode-on-resume --allow-reward-change --additional-cycles 5000
```

Git只提供代码、固定训练实例及溯源清单，不包含模型参数、训练检查点和历史结果。
本地完整交付包保留基础运行时；从Git克隆后需另行复制`inference_assets/runtime.pt`。
不要把训练checkpoint直接当推理runtime；先用下方导出入口。GPU长跑需在目标硬件验证。

## 结构与验证

- `scripts/train_e2e_single.py`：全实例同步采样与GRPO更新。
- `src/causal_schedule_lab/m3/`：模型、候选构造、执行、奖励与批量更新。
- `scripts/build_train128.py`：从可信旧训练检查点重建固定128集合。
- `scripts/export_e2e_runtime.py`、`run_fast_parallel.py`：冻结模型导出与并行搜索。
- `VALIDATION_V13.md`：本版本测试边界。历史版本说明如下，仅供追溯；当前配置以上述V13为准。

## V12当前规则（优先于下方历史版本说明）

### 训练与推理

训练使用scripts/train_e2e_single.py或run_e2e_single.sh。
训练检查点不能直接传给--runtime；先运行：

```bash
python scripts/export_e2e_runtime.py --checkpoint /path/to/latest.pt --output /path/to/trained_runtime.pt
python scripts/run_fast_parallel.py --runtime /path/to/trained_runtime.pt --output /path/to/new_results --workers 16 --branches 16 --batches 50 --horizon 10
```

导出自动识别L4/L6并严格加载全部活跃模块。推理初始化同步候选回溯和传播深度，
加载一次冻结网络，默认单+双算子；--single-only为可选消融。不要省略--runtime而误用包内旧模型。
推理从--bank中的调度开始，训练默认从规则解开始；推理的多状态探索不等同于训练500步回合重置。
负载辅助项只参与训练信用；推理保存最好解只比较工期。

- 每次采集/更新覆盖全部26张TRAIN图，16轨迹/图，共416条；16 worker分批执行。
  不再轮换16张子集；测试时显式--instances可缩小集合。
- 每个大回合每张图分配500步；每批次统一10步或20步（任一实例停滞2批次则统一20步）。
  所有图预算一起达到500再一起恢复各自固定的初始规则解。网络/优化器持续训练。
  早停轨迹仍按分配预算计数，不保证实际执行500个有效动作。不是500个大回合。
- 从V11升级须--reset-episode-on-resume --allow-reward-change：保留权重、优化器、
  全局最好解，但明确开始新回合，不延续旧16图预算。V12以后正常续训不加reset。
- 负载成本 B=max_m(load_m)+0.1*sum_m(load_m)，load是加工时长之和，不是机器完工时刻。
  默认辅助权重0.2；轨迹辅助回报=clip(0.2*(B_start-B_terminal), ±0.05*C_start)。
  以上是原始辅助项，随后按每图轨迹组校准：sum(abs(load))/sum(abs(main)+abs(load))=35%。
  校准发生在GRPO标准化之前，最终辅助项不再受上述5%原始限幅约束。
  每步未来信用也按同一局部步序的兄弟轨迹单独校准到35%。正负号保留。
  主项非零但负载不变时占比0%；主项全零而负载有变化时占比100%；不虚构信号。
  这是奖励幅值比例，不是梯度贡献比例。LOAD_SHARE默认0.35，可以配置；0关闭辅助。
  不逐步累加绝对负载奖励，序列移动不改变加工负载时辅助值为0。
  这是启发式多目标塑形，不保证最优策略不变；小权重经GRPO标准化仍可能主导工期打平组。
- lambda=0.2工期回退惩罚保持；最好调度/成功标记仍只按工期判断，不按混合奖励选。
- collect日志记录校准前原始分量；[reward-mix]显示校准后的组内比例。
  reward_components.jsonl逐图即时保存最终main/load/total，TensorBoard分别记录均值。
  LOAD_WEIGHT=0可做对照；改变奖励续训须显式--allow-reward-change。
- 微批次16、epoch1、单/双算子、L6均保留。更多图增加每批总工作量，GPU峰值仍需实测。

以下是合并的历史修改说明，版本号及旧启动示例不代表当前推荐命令。

本文件是当前版本说明。README_V6/V7/V8 等文件仅是历史记录，不能用其旧参数启动。
本完整包已合并 V8 的最佳解续接、500步回合、回放校验、严格FP32、根因/动作扰动、
内存轨迹传输与安全保存补丁；不需要再依次安装历史补丁。

## V11新增：L=4到L=6检查点迁移

- 默认trace-hops=6，同步设置候选上游BFS、传播子图的最大范围、ProbabilityTrace.hops。
  L不是轨迹步数，也不是GAT/Transformer层数。原有0～4深度仍参加融合，新增5、6深度。
- 两处函数的默认参数由静态4改为运行时读取，避免只改配置却仍按4跳建图。
- 旧层数嵌入前4行、深度评分前5行以及全部共享边/种子/其他网络权重保持不变。
  新增嵌入复制最后一行；新增深度评分复制最后一行并将偏置减4，使新增深度
  初始logit低于原最深层。不是把新增深度固定关闭，后续可以学习。
- 仅形状改变的hop_embedding.weight、depth.weight、depth.bias三组参数重置Adam状态；
  其他参数的优化器状态保留。L=6检查点再次续训不重复重置任何参数。
- 启动输出[trace]和[trace migration]，并在输出目录保存trace_migration.json。
- 深度扩大及hop特征归一化会改变行为，不能宣称与L=4输出完全相同。
  只从新采样轨迹更新；旧调度、回合计数、最好状态及根因/算子历史保留。
- GPU微批次默认改为16，workers=16、branches=16保持不变。L=6可能增加图规模、显存与耗时，
  CPU测试不保证4090不会OOM。
- 本次增量包基于已安装V10的目录；完整V11包已包含全部历次修改。

## V10新增：停滞后的长探索和安全重启

- 每个实例独立统计连续没有刷新“本回合最好工期”的批次数。
- 连续2批次不改善：下一次轨迹由10步增至20步；改善后恢复10步。
- 连续3批次不改善：尝试从最好状态的候选池取一个不同调度作为临时探索起点。
  候选必须是直接从当前最好状态采样得到，且工期不超过它的110%；不会把已扰动状态
  后续生成的更差状态加入重启池。因此下次换状态仍以最好状态为共同起点，不连续漂移。
  找不到合法的池内候选就继续从最好状态探索，不做强制随机破坏。
- 一次重启之后，普通批次回到最好状态；再次重启至少隔3个未改善批次。
  重启本身不清零停滞计数，不会意外把20步降回10步。只有真正改善才清零。
- 最好状态改变时清空旧重启池并重建；V8/V9缺少来源标签的旧池在续训时作废，
  但模型、优化器、最好调度、根因/算子历史及回合进度保留。
- 所选实例每回合各500步“分配预算”，按实际分配的10/20步累计；提前终止仍消耗
  本次已分配预算（延续旧语义），最后一批按剩余预算截短。不是机械执行50次更新。
  同一回合中先达到500步的实例暂停采样，等其余实例也完成后一起进入下一回合。
- checkpoint保存episode_clock；旧版通过累计cycle与episode_origin_cycle恢复原进度。
  例如cycle=50、origin=13对应已用370步，不会错误地重置或补算成500步。
- gain仍是相对临时起点的轨迹收益；新增incumbent_gain表示是否真正突破最好状态。
  不把从较差扰动状态恢复到旧最好状态误报为全局改善。

## 本次修改

- 单算子保留；额外生成有界双算子候选。每个原子修改最多参与一个组合，不强行配对。
- 旧200实例检查点（498次更新）的留存记录发现机器迁入/迁出衔接、共同迁出源机器、
  跨机器工序联系等成功案例。因此在旧的依赖/机器邻接候选上增加共同源机器和
  双向迁入迁出联系，不新增算子家族、不改变合法性检查。
- 搭档按结构关系信号数排序，平局沿用已有单算子预测分数和稳定签名排序；最多
  检查8个搭档。结构信号排序是待验证启发式，不是通过全部训练尝试拟合的最优权重。
  不把只出现过成功案例等同于有正期望收益，不把联合收益称为协同增益。
- 旧数据是保留路径，有筛选偏差；尚未完成588条可重放样本的A/B/AB反事实评估。
  本版不是那项反事实评估的结论。见 evidence/old200_pair_relationship_audit.json。
- lambda默认0.2：start-best - 0.2*(terminal-best) - 原有不可行惩罚。
  保留后缀奖励/优势分配逻辑；不是新增每步硬门槛，允许暂时变差。
- rollout_metrics.jsonl每完成一条轨迹即写入：逐动作签名、家族、前后工期、相对变化、
  执行结果，以及末端退化和轨迹奖励。不保存大型特征张量；失败缺失值为null，不伪造0收益。

## 保持不变

26实例规则初始调度；SFT初始化，参与前向计算的模块端到端RL；无KL、无anchor；
每回合选16实例、每实例16轨迹、16 workers、epoch=1；V11 decision batch默认16。
每回合500步，回合内保留并从最好状态续接，回合结束重置所选实例调度；历史最优独立保存。
根因/算子软扰动保留；换状态扰动默认开启（perturb-after=3），带来源约束。
所有实例不会在同一回合同时采样：所选16实例连续执行一个回合，下一回合轮换。

## 续训

停止旧进程并备份代码后，增量包可以直接解压到已有 /root/t2m_e2e_dispatch_v8。
完整包可以解压到独立目录，它包含代码、固定26实例训练清单所在数据银行和启动资产，
不包含服务器上正在训练的latest.pt；继续训练需指定那个最新检查点。

```bash
export OUTPUT=/root/autodl-tmp/causasched_runs/v11_$(date +%Y%m%d_%H%M%S)
mkdir -p "$(dirname "$OUTPUT")"
export WORKERS=16 BRANCHES=16 ROOTS_PER_CYCLE=16 DECISION_BATCH=16 EPOCHS=1
bash run_e2e_single.sh --resume /absolute/path/latest.pt \
  --allow-reward-change --regression-weight 0.2 \
  --trace-hops 6 --adaptive-after 2 --long-horizon 20 --perturb-after 3 --additional-cycles 5000
```

默认lambda=0.2。旧lambda不同时必须明确传allow-reward-change；保留权重、优化器及回合进度。
候选集改变只用于新采样轨迹，不复用旧轨迹做离策略更新。不传reset-episode-on-resume时
不会从初始调度重启当前回合。5000是新增更新批次数，不是新增5000个回合。
20步会增加该批次采样和更新耗时、内存需求，并非免费增加深度。

## 检查

```bash
python scripts/test_v9_relations.py
python scripts/test_v10_adaptive.py
python scripts/test_v11_trace6.py
python scripts/check_v6_batch.py --device cpu
```

本地CPU功能测试不是AutoDL的CUDA速度/收敛测试；不保证硬件利用率或效果提升。
更改候选规则后应将结果与旧版本分目录记录。原有训练数据、检查点及旧版本包不删除。
