# pi0 LIBERO SFT 使用说明

这个目录提供一个最小化的 pi0 SFT 流程，分为三步：

1. 从任意目录读取 HDF5 数据，转成 LeRobot 数据，并保存到任意目录。
2. 从任意目录读取 LeRobot 数据，训练模型，并保存到任意模型目录。
3. 从任意模型目录读取 checkpoint，启动模型服务，并用另一个进程跑 LIBERO 测试。

核心约定只有三个路径：

```bash
HDF5_DIR=/path/to/hdf5_demos
LEROBOT_DATA_DIR=/path/to/lerobot_dataset
MODEL_DIR=/path/to/model_output
```

训练结束后，模型目录结构是：

```text
${MODEL_DIR}/
  steps/
    999/
    1999/
    2999/
  final/
```

`steps/` 下面是正常训练过程中保存的 step checkpoint；`final/` 是训练结束后自动发布的最终模型目录。测试时既可以读取 `final/`，也可以读取某个具体 step。

## 1. 准备数据

脚本：

```bash
bash train/prepare_pi0_libero_data.sh
```

它做两件事：

1. 从 `HDF5_DIR` 递归读取 `.hdf5` / `.h5` 文件。
2. 将数据转换成 OpenPI 可读取的 LeRobot 格式，写入 `LEROBOT_DATA_DIR`。
3. 为这份 LeRobot 数据计算归一化统计量，默认写入 `${LEROBOT_DATA_DIR}/openpi_assets`。

转换时会将 `agentview_rgb` 和 `eye_in_hand_rgb` 旋转 180 度，以匹配 OpenPI 官方 LIBERO eval 中的图像预处理。动作直接使用 HDF5 里的 `actions` 字段，不在 converter 中做 state 差分。

示例：

```bash
cd /data/huangdi/heliqun/pi0

HDF5_DIR=/data/aoss/heliqun/dataset/libero-dataset/libero_goal_select_50 \
LEROBOT_DATA_DIR=/data/aoss/heliqun/lerobot/libero_goal_select_50 \
bash train/prepare_pi0_libero_data.sh
```

准备完成后，数据目录大致是：

```text
/data/aoss/heliqun/lerobot/libero_goal_select_50/
  data/
  meta/
  openpi_assets/
```

其中：

```text
data/ 和 meta/         LeRobot 数据本体
openpi_assets/         训练需要的 norm stats
```

常用参数：

```bash
HDF5_DIR=/path/to/hdf5_demos          # 输入 HDF5 数据目录
LEROBOT_DATA_DIR=/path/to/lerobot     # 输出 LeRobot 数据目录
OVERWRITE_LEROBOT=true                # 目标 LeRobot 数据存在时是否覆盖
MAX_EPISODES=10                       # 只转换前 N 个 episode，用于快速检查
COMPUTE_NORM_STATS=true               # 是否计算 norm stats
NORM_MAX_FRAMES=2048                  # 只用部分 frame 计算 norm stats，用于快速检查
STRICT_DATA=true                      # 严格要求 wrist image 和 8D state 字段
```

默认情况下，`STRICT_DATA=false`。如果 HDF5 中没有 wrist image 或 8D state，转换脚本会填充 zero fallback，让流程可以先跑通。

## 2. 训练模型

脚本：

```bash
bash train/train_pi0_libero_sft.sh
```

它只做训练，不做数据转换。训练时从 `LEROBOT_DATA_DIR` 读取 LeRobot 数据和 norm stats，然后将模型保存到 `MODEL_DIR`。

示例：

```bash
cd /data/huangdi/heliqun/pi0

LEROBOT_DATA_DIR=/data/aoss/heliqun/lerobot/libero_goal_select_50 \
MODEL_DIR=/data/aoss/heliqun/models/pi0/libero_goal_select_50_sft \
NUM_TRAIN_STEPS=3000 \
BATCH_SIZE=16 \
WANDB_ENABLED=false \
bash train/train_pi0_libero_sft.sh
```

训练输出：

```text
/data/aoss/heliqun/models/pi0/libero_goal_select_50_sft/
  steps/
    999/
    1999/
    2999/
  final/
```

`steps/` 是训练过程中保存的 checkpoint。`final/` 是训练结束时自动从最后一个 step 发布出来的最终可加载模型。

常用训练参数：

```bash
LEROBOT_DATA_DIR=/path/to/lerobot_dataset
MODEL_DIR=/path/to/model_output
NUM_TRAIN_STEPS=3000
BATCH_SIZE=16
NUM_WORKERS=4
SAVE_INTERVAL=1000
LOG_INTERVAL=100
SEED=42
WANDB_ENABLED=false
OVERWRITE=true       # 如果 MODEL_DIR/steps 已存在，覆盖旧训练 checkpoint
RESUME=true          # 从 MODEL_DIR/steps 中恢复训练
```

注意：

```text
OVERWRITE=true 和 RESUME=true 不要同时使用。
```

训练脚本会检查：

```text
${LEROBOT_DATA_DIR}/data
${LEROBOT_DATA_DIR}/meta
${LEROBOT_DATA_DIR}/openpi_assets
```

如果你只想打印训练命令，不真正启动训练：

```bash
LEROBOT_DATA_DIR=/path/to/lerobot_dataset \
MODEL_DIR=/path/to/model_output \
DRY_RUN=true \
WANDB_ENABLED=false \
bash train/train_pi0_libero_sft.sh
```

## 3. 测试和部署

测试脚本是独立入口，不依赖训练目录下的 config。

脚本：

```bash
bash test/test.sh
```

测试需要两个进程：

1. 一个进程部署模型服务。
2. 另一个进程连接这个服务并跑 LIBERO 测试。

### 3.1 启动模型服务

读取最终模型：

```bash
cd /data/huangdi/heliqun/pi0

MODEL_PATH=/data/aoss/heliqun/models/pi0/libero_goal_select_50_sft/final \
MODE=serve \
PORT=8000 \
bash test/test.sh
```

读取某个 step：

```bash
MODEL_PATH=/data/aoss/heliqun/models/pi0/libero_goal_select_50_sft/steps/2999 \
MODE=serve \
PORT=8000 \
bash test/test.sh
```

`MODEL_PATH` 必须指向一个可加载 checkpoint 目录，目录下需要有：

```text
params/
assets/
```

### 3.2 启动 LIBERO 测试

另开一个终端：

```bash
cd /data/huangdi/heliqun/pi0

MODE=eval \
HOST=0.0.0.0 \
PORT=8000 \
TASK_SUITE_NAME=libero_goal \
TASK_ID=6 \
NUM_TRIALS_PER_TASK=1 \
bash test/test.sh
```

常用测试参数：

```bash
MODEL_PATH=/path/to/model/final       # 只在 MODE=serve 时需要
CONFIG_NAME=pi0_libero_low_mem_finetune
PORT=8000
HOST=0.0.0.0
TASK_SUITE_NAME=libero_goal           # libero_spatial / libero_object / libero_goal / libero_10 / libero_90
TASK_ID=6                             # -1 表示跑整个 suite
NUM_TRIALS_PER_TASK=50
VIDEO_OUT_PATH=/path/to/videos
```

只打印命令、不真正启动服务或测试：

```bash
MODEL_PATH=/path/to/model/final MODE=serve DRY_RUN=true bash test/test.sh
MODE=eval DRY_RUN=true NUM_TRIALS_PER_TASK=1 bash test/test.sh
```

## 4. 推荐的完整命令

准备数据：

```bash
HDF5_DIR=/data/aoss/heliqun/dataset/libero-dataset/libero_goal_select_50 \
LEROBOT_DATA_DIR=/data/aoss/heliqun/lerobot/libero_goal_select_50 \
bash train/prepare_pi0_libero_data.sh
```

训练：

```bash
LEROBOT_DATA_DIR=/data/aoss/heliqun/lerobot/libero_goal_select_50 \
MODEL_DIR=/data/aoss/heliqun/models/pi0/libero_goal_select_50_sft \
NUM_TRAIN_STEPS=3000 \
BATCH_SIZE=16 \
WANDB_ENABLED=false \
bash train/train_pi0_libero_sft.sh
```

部署模型：

```bash
MODEL_PATH=/data/huangdi/heliqun/pi0/sft_runs/model/final \
MODE=serve \
PORT=8000 \
bash test/test.sh
```

运行测试：

```bash
MODE=eval \
HOST=0.0.0.0 \
PORT=8000 \
TASK_SUITE_NAME=libero_goal \
TASK_ID=6 \
NUM_TRIALS_PER_TASK=10 \
bash test/test.sh
```

## 5. 缓存说明

JAX/XLA 和 CUDA cache 默认放在 pi0 根目录下：

```text
/data/huangdi/heliqun/pi0/cache/
  jax/
  cuda/
```

这些 cache 不放在每次 run 的输出目录下，因此换 `LEROBOT_DATA_DIR` 或 `MODEL_DIR` 后仍然可以复用。只要模型结构、shape、JAX/CUDA 版本和 GPU 架构一致，后续运行可以减少重复编译时间。
