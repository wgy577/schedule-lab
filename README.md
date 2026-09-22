# CausaSched 使用说明

## 1. 安装

```bash
git clone https://github.com/wgy577/schedule-lab.git
cd schedule-lab/experiments/causasched

# 先安装与服务器CUDA匹配的PyTorch，再安装其余依赖
pip install -r requirements_e2e.txt
```

将基础运行时放到 `inference_assets/runtime.pt`。该文件不在Git中，可从完整运行包复制；使用完整包时直接进入解压目录即可。

## 2. 开始训练

默认使用固定128个训练实例，每次更新全部实例。每张图每个大回合200步预算，完成后恢复各自固定的初始规则解；网络和优化器持续学习。小段通常10步，停滞时统一增加到20步；提前结束仍计入分配预算。

默认16个worker、每图16条轨迹、GPU微批次16、每次更新1个epoch、回溯深度L=6，启用单算子和双算子。负载辅助奖励的组内幅值目标占比为35%；没有负载变化时为0。最好解只按工期保存。

```bash
export WORKERS=16
export BRANCHES=16
export DECISION_BATCH=16
export EPOCHS=1

export OUTPUT="/root/autodl-tmp/causasched_runs/train128_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$(dirname "$OUTPUT")"

nohup bash run_e2e_single.sh --cycles 5000 > "${OUTPUT}.log" 2>&1 &
echo $! | tee "${OUTPUT}.pid"
tail -f "${OUTPUT}.log"
```

`--cycles 5000`表示5000次采集更新，不是5000个大回合。非AutoDL环境请自行修改输出路径；显存不足时可降低 `DECISION_BATCH`。

## 3. 当前128实例版本普通续训

不加重置回合参数，继续已保存的回合进度。

```bash
RESUME="/path/to/train128/latest.pt"
export OUTPUT="/root/autodl-tmp/causasched_runs/train128_continue_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$(dirname "$OUTPUT")"

nohup bash run_e2e_single.sh \
  --resume "$RESUME" --additional-cycles 5000 \
  > "${OUTPUT}.log" 2>&1 &
echo $! | tee "${OUTPUT}.pid"
tail -f "${OUTPUT}.log"
```

## 4. 查看日志与停止训练

终端默认只输出阶段与批次总结；启动时追加 `--verbose` 可恢复详细输出。重新打开终端后，请用实际路径替换下列路径。

```bash
tail -f /path/to/run.log
tensorboard --logdir /path/to/run/tensorboard --host 0.0.0.0 --port 6007

# 停止当前项目的训练进程，不删除结果
python scripts/stop_project_training.py
```

输出目录包含：

- `latest.pt`：训练续训检查点。
- `detail.log`：详细父进程日志。
- `rollout_metrics.jsonl`：轨迹、工期和提前结束原因。
- `reward_components.jsonl`：最终工期奖励与负载奖励分量。
- `audit_*.json`、`tensorboard/`：每次更新的统计。
- `config.json`、`initial_manifest.json`：实际配置及初始实例信息。

## 5. 导出并运行推理

训练检查点不能直接作为推理的 `--runtime`；先导出一次：

```bash
python scripts/export_e2e_runtime.py \
  --checkpoint /path/to/latest.pt \
  --output inference_assets/trained_runtime.pt

python scripts/run_fast_parallel.py \
  --runtime inference_assets/trained_runtime.pt \
  --bank data/train128 \
  --output outputs/inference_run_01 \
  --workers 16 --branches 16 --batches 50 --horizon 10
```

导出文件及推理输出目录使用新路径，避免覆盖已有结果。推理冻结参数，从指定bank内的调度开始搜索；默认启用单、双算子，追加 `--single-only` 可只用单算子。快速推理入口使用CPU，每个worker只加载一次运行时。

上述命令在128个训练实例上做功能检查，不能作为未见测试结果。正式测试请通过 `--bank` 指定外部独立实例集。仓库仅保留这128个实例及其加载清单。
