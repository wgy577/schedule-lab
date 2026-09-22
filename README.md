# CausaSched 使用说明

## 安装

```bash
git clone https://github.com/wgy577/schedule-lab.git
cd schedule-lab/experiments/causasched
# 先安装与CUDA匹配的PyTorch
pip install -r requirements_e2e.txt
```

将完整运行包的 `runtime.pt` 放到 `inference_assets/runtime.pt`。运行时、模型权重和个人实验结果不在Git中，仅克隆仓库不能直接启动训练。

## 训练

当前代码使用多轨迹候选比较：每图16条轨迹，每步按策略无放回抽取最多40个不同候选，执行其中工期最小的可行动作；候选不足时全部比较。允许暂时恶化，单、双算子均可参与。M3更新使用整组有序抽样的联合对数概率，不使用获胜动作的单次采样概率。

每个大回合200步，结束后回到固定初始规则解；小段通常10步，停滞时可增加到20步。停滞换状态已关闭，根因和算子选择探索保留。每次更新1个epoch，网络端到端训练，不使用SFT参考KL。

即时工期奖励为正改善减去0.2倍恶化量；辅助项默认是0.5倍整体机器加工负载标准差的下降量，不做奖励占比缩放。不可行执行另受惩罚；组相对优势仍标准化。

仓库保留原有128个训练实例；服务器26实例训练需要原完整包中的实例bank和检查点，未额外上传数据。

```bash
# 仓库128实例训练入口
export WORKERS=16 BRANCHES=16 EPOCHS=1 DECISION_BATCH=16
export E2E_STEP_CANDIDATES=40
export OUTPUT="outputs/train40_$(date +%Y%m%d_%H%M%S)"
mkdir -p outputs
python scripts/test_step20_training.py
nohup bash run_e2e_single.sh --cycles 5000 > "${OUTPUT}.log" 2>&1 &
echo $! | tee "${OUTPUT}.pid"
tail -f "${OUTPUT}.log"
```

`--cycles` 是采集更新次数，不是大回合数。`test_step20_training.py` 沿用旧文件名，测试候选组概率和执行逻辑，不限定运行时只能比较20个。

已有26实例运行包，使用明确的检查点续训：

```bash
python scripts/start_step40_training.py \
  --resume /path/to/current/latest.pt \
  --output outputs/continue40 \
  --additional-cycles 5000
```

该入口要求原检查点为26实例、16轨迹、200步回合，保留其奖励系数、权重、优化器和当前状态；固定16个worker、每步最多40候选。启动打印实例ID与问题指纹，并写入 `resumed_cohort.json`。请勿将训练实例用作独立测试证据。

## 日志与停止

```bash
tail -f /path/to/run.log
tensorboard --logdir /path/to/run/tensorboard --host 0.0.0.0 --port 6007
python scripts/stop_project_training.py
```

`[candidate-set]` 的 `trials/decisions` 是每步实际比较数量；`trial_pairs` 是试过的双算子数量，`executed_pairs` 是最终选中的数量。`[timing]` 区分采集、准备与更新耗时。`latest.pt` 保存续训状态，诊断日志逐批写入。停止脚本不删除结果。

## 推理

```bash
python scripts/export_e2e_runtime.py \
  --checkpoint /path/to/latest.pt \
  --output inference_assets/trained_runtime.pt
python scripts/run_fast_parallel.py \
  --runtime inference_assets/trained_runtime.pt \
  --bank data/train128 --output outputs/inference_check \
  --workers 16 --branches 16 --batches 50 --horizon 10
```

上述快速推理入口仍是每步采样一个动作，不是40候选比较训练入口。它冻结参数、每个worker只加载一次运行时。上述bank仅用于训练集功能检查；正式测试请指定独立实例bank。
