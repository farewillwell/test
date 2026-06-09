# ours: pi0 iterative AWBC + IQL + QSelect

这个目录只放我们自己的迭代学习代码，不修改 `train/`、`test/`、`openpi/`，也不修改
`/data/huangdi/heliqun/openvla-oft/openvla-oft/iter-learn`。

当前实现的目标很单一：

- 输入数据从一开始就是 LeRobot。
- IQL critic 从任意 LeRobot 目录读取 episode，并用成功/失败 metadata 作为 reward。
- Q/advantage label 写到独立目录。
- AWBC 数据通过 advantage 对 episode 重采样，输出新的 LeRobot 目录。
- AWBC 训练从上一轮 policy 的 `params/` 初始化，保存到本轮 `model/steps/` 和 `model/final/`。
- 在线 collect 开两个进程：
  - `pi_env` 进程：加载 OpenPI policy + IQL critic，生成多个 action sample，并用 qselect 选 action chunk。
  - `libero_env` 进程：只负责 LIBERO 环境、执行动作、把 observation/action 直接写成 LeRobot。
- `iter.py` 使用 `save.json` 做 stage 级恢复。

## 推荐入口

先保证 `libero_env` 能写 LeRobot：

```bash
cd /data/huangdi/heliqun/pi0
uv pip install \
  --python openpi/examples/libero/libero_env/bin/python \
  -r ours/requirements_libero_writer.txt
```

然后用一个入口脚本跑完整迭代：

```bash
cd /data/huangdi/heliqun/pi0

export TASK_ID=6
export TASK_SUITE_NAME=libero_goal
export RUN_NAME=goal-${TASK_ID}-qselect
export WORKSPACE=/data/aoss/heliqun/pi0-ours/${RUN_NAME}

export SEED_DATA_DIR=/data/huangdi/heliqun/pi0/sft_runs/lerobot_data
export INIT_MODEL_DIR=/data/huangdi/heliqun/pi0/sft_runs/model/final

export ITERS=4
export GPUS=2
export BATCH=16
export TRAJ_COUNT_PER_TASK=50
export IQL_STEPS=4000
export TRAIN_STEPS=3000
export NUM_ACTION_SAMPLES=16
export SAMPLE_MODE=qselect

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export LIBERO_CONFIG_PATH=/data/huangdi/heliqun/openvla-oft/openvla-oft/LIBERO/.libero
export MUJOCO_GL=egl

bash ours/run_iter.sh
```

看命令不真跑：

```bash
DRY_RUN=true bash ours/run_iter.sh
```

恢复也用同一条命令。脚本会读取 `${WORKSPACE}/save.json`，从断掉的 stage 继续。

## 目录结构

一次迭代会写到：

```text
workspace/
  save.json
  iter0/
    q/final.pt
    labels/labels.jsonl
    awbc_lerobot/
    model/
      steps/
      final/
    collected_lerobot/
```

下一轮默认使用上一轮 `collected_lerobot/` 作为 IQL/AWBC 输入数据，并使用上一轮 `model/final/` 作为 policy 初始化。

## 环境依赖

模型推理和 qselect 使用：

```bash
/data/huangdi/heliqun/pi0/openpi/pi_env/bin/python
```

LIBERO 环境和轨迹收集使用：

```bash
/data/huangdi/heliqun/pi0/openpi/examples/libero/libero_env/bin/python
```

因为 collector 要在 `libero_env` 里直接写 LeRobot，所以这个 env 需要能 import `lerobot/pandas/pyarrow`。当前脚本会在启动模型服务前检查；缺依赖时按下面装：

```bash
cd /data/huangdi/heliqun/pi0
uv pip install \
  --python openpi/examples/libero/libero_env/bin/python \
  -r ours/requirements_libero_writer.txt
```

## 完整迭代

```bash
cd /data/huangdi/heliqun/pi0

openpi/pi_env/bin/python ours/iter.py \
  --workspace /data/aoss/heliqun/pi0_ours/run1 \
  --seed-data-dir /path/to/seed_lerobot \
  --init-model-dir /path/to/init_model/final \
  --iters 4 \
  --iql-steps 4000 \
  --train-steps 3000 \
  --num-action-samples 8 \
  --sample-mode qselect \
  --task-suite-name libero_goal \
  --task-id 6 \
  --num-trials-per-task 50
```

如果中途断了，重新执行同一条命令即可。`save.json` 会记录当前 `iter`、`next_stage`、`data_dir`、`policy_dir`。例如断在 `collect`，恢复时只重跑 `collect`。

查看 dry-run 展开的实际命令：

```bash
openpi/pi_env/bin/python ours/iter.py \
  --workspace /tmp/pi0_ours_iter_dry \
  --seed-data-dir /data/huangdi/heliqun/pi0/sft_runs/lerobot_data \
  --init-model-dir /data/huangdi/heliqun/pi0/sft_runs/model/final \
  --iters 1 \
  --iql-steps 2 \
  --train-steps 2 \
  --num-trials-per-task 1 \
  --dry-run
```

## 单独训练 IQL

```bash
openpi/pi_env/bin/python ours/light_iql.py \
  --data-dir /path/to/lerobot \
  --save-dir /path/to/q \
  --views image,wrist_image \
  --horizon 10 \
  --hidden-dim 512 \
  --num-q 2 \
  --max-steps 4000 \
  --default-success
```

critic 结构包含 CLIP 图文编码、action chunk transformer、Q ensemble 和 V head。

## 单独打 Q/adv 标签

```bash
openpi/pi_env/bin/python ours/label_q.py \
  --data-dir /path/to/lerobot \
  --critic-path /path/to/q/final.pt \
  --output-dir /path/to/labels
```

输出：

```text
labels/labels.jsonl
labels/summary.jsonl
```

## 构造 AWBC LeRobot 数据

```bash
openpi/pi_env/bin/python ours/make_awbc_data.py \
  --src-data-dir /path/to/lerobot \
  --label-dir /path/to/labels \
  --output-dir /path/to/awbc_lerobot
```

现在的 AWBC 是 episode 级 advantage 重采样，因为我们不改 OpenPI 训练 dataloader，所以没有把 per-frame weight 塞进训练 loss。

## 单独启动 qselect policy server

这个进程在 `pi_env` 里运行，负责模型推理和 qselect：

```bash
cd /data/huangdi/heliqun/pi0/openpi

/data/huangdi/heliqun/pi0/openpi/pi_env/bin/python -u /data/huangdi/heliqun/pi0/ours/qselect_policy_server.py \
  --config pi0_libero_low_mem_finetune \
  --checkpoint-dir /path/to/model/final \
  --critic-path /path/to/q/final.pt \
  --host 0.0.0.0 \
  --port 8000 \
  --num-action-samples 8 \
  --sample-mode qselect
```

返回结果包含：

- `actions`: 被选中的 action chunk。
- `candidate_actions`: 所有候选 action chunk。
- `selected_index`: 被选中的候选编号。
- `q_values`: critic 给每个候选的 Q 分数。

## 单独 collect 并写 LeRobot

最简单是用脚本同时启动 server 和 collector：

```bash
cd /data/huangdi/heliqun/pi0

MODEL_PATH=/path/to/model/final \
CRITIC_PATH=/path/to/q/final.pt \
OUTPUT_DIR=/path/to/collected_lerobot \
CONFIG_NAME=pi0_libero_low_mem_finetune \
NUM_ACTION_SAMPLES=8 \
SAMPLE_MODE=qselect \
TASK_SUITE_NAME=libero_goal \
TASK_ID=6 \
NUM_TRIALS_PER_TASK=50 \
bash ours/run_collect_with_server.sh
```

collector 写出的 LeRobot 数据会额外包含：

```text
meta/ours_rollouts.jsonl
```

这里记录 `success/reward/task_id/rollout_index/length`。IQL 会读取这个文件来区分成功和失败轨迹。
