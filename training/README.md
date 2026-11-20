# 双臂 22D 策略训练与离线评估

以下命令从仓库根目录执行。训练使用 `third_party/groot_train` 的 πR² 分支固定提交。两种相机配置共用绝对关节目标、当前帧状态和 16 步预测窗口。输出关节顺序为左臂 7、右臂 7、左手 4、右手 4。语言来自每一帧的 `task_index`。

## 准备环境和输入

参考训练平台为 Linux x86_64、支持 BF16 的 NVIDIA GPU、CUDA 12.8 和 Python 3.10。系统需提供 CUDA 工具链、FFmpeg 共享库及 libaio 开发文件。上游锁文件固定 PyTorch 2.7.1、TorchCodec 0.4.0、Transformers 4.57.3 和 DeepSpeed 0.17.6。已具备系统依赖和 uv 时执行以下命令。

```bash
bash scripts/setup_sources.sh
(cd third_party/groot_train && uv sync --frozen --python 3.10)
source third_party/groot_train/.venv/bin/activate
python scripts/prepare_weights.py check base
python scripts/prepare_weights.py check backbone
```

训练目录默认是 `data/processed/train_view`，其中的 `meta/stats.json` 仅由训练帧计算。原始 LeRobot v3 数据的转换与切分见[数据处理](../data_pipeline/README.md)。模型目录需分别包含完整的 GR00T 和 Cosmos 快照，文件清单见[权重说明](../weights/README.md)。

## 选择训练配方

| 配置 | 相机 | GPU 数 | 优化器步数 | 单次全局批量 × 累积 | 预热比例 | 保存间隔 / 保留数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `headrgb pir2` | 头部 | 2 | 317000 | 256 × 1 | 0.0078864 | 15000 / 10 |
| `rgb3 pir2` | 头部、左腕、右腕 | 2 | 50000 | 64 × 2 | 0.05 | 1000 / 15 |

这两个预设取自已保存的 16 小时头部 RGB 实验和三相机 50k 实验配置。17 小时三相机数据处理流程输出兼容 `rgb3` 的格式。在新数据上运行 50k 配方会得到新实验。两种配方的初始学习率均为 1e-4，优化器为 AdamW，weight decay 为 1e-5，种子为 42。双卡默认 ZeRO-2，冻结视觉与语言骨干，微调投影和动作专家。每个优化器步的样本数为全局批量乘以梯度累积次数。

```bash
# 在两卡机器上启动三相机实验。换成 headrgb 即使用头部相机配方。
CUDA_VISIBLE_DEVICES=0,1 bash training/train.sh rgb3 pir2

# 将同一配方的动作专家改为标准 flow matching 基线
CUDA_VISIBLE_DEVICES=0,1 bash training/train.sh rgb3 plain_flow
```

πR² 使用 0.2 恒定时间步与 0.8 分块时间步混合，最大块大小 5，屏蔽已去噪端的损失。模型图像延迟上限为 5 个控制周期，延迟嵌入维度 64。数据加载器通过 `GR00T_IMAGE_DELAY_MAX` 单独采样帧延迟。三相机原脚本和本仓库 πR² 默认都显式设为 5。317k 原脚本依赖调用 shell，历史环境值无法从检查点恢复；模拟当时未设置该变量的情况可用 `GR00T_IMAGE_DELAY_MAX=0` 启动。标准 flow 基线把图像延迟设为 0。

## 小规模验证、断点与产物

先在至少包含一个完整 16 帧窗口的训练集上做一步冒烟测试，运行目录与正式实验分开。模型仍会保存完整检查点和最终权重，请预留数十 GB 磁盘。原 317k 训练脚本按保留十个完整续训检查点预算 450 GB。

```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 GLOBAL_BATCH_SIZE=1 ACCUMULATION_STEPS=1 \
  MAX_STEPS=1 SAVE_STEPS=1 WORKERS=1 NUM_SHARDS_PER_EPOCH=2 \
  OUTPUT_DIR=outputs/smoke_head bash training/train.sh headrgb pir2

python scripts/prepare_weights.py check checkpoint --path outputs/smoke_head/checkpoint-1
```

`bash training/train.sh --help` 列出参数。数据和权重路径可以通过 `DATASET`、`BASE_MODEL` 和 `GR00T_COSMOS_LOCAL_PATH` 覆盖。对较小显存的卡降低全局批量，同时增加梯度累积次数即可维持每步样本数。原三相机 20k 配方可在 `rgb3 pir2` 上设置 `NUM_GPUS=4 MAX_STEPS=20000 SAVE_TOTAL_LIMIT=10`，其他参数与三相机表项一致。原单相机 240k 启动配方可设置 `MAX_STEPS=240000 GLOBAL_BATCH_SIZE=64 ACCUMULATION_STEPS=2 WARMUP_RATIO=0.0104167 SAVE_STEPS=20000 WORKERS=2`。每次都使用独立的 `OUTPUT_DIR`。

默认输出在 `outputs/<相机配置>_<策略>/`。其中 `launch.log` 留存实际命令、数据延迟和运行日志，`experiment_cfg/conf.yaml` 留存展开的配置。用于推理的 `checkpoint-N/` 需要同时保留配置、模型所有分片、处理器、归一化统计量和 embodiment ID。续训还需要优化器、调度器、Trainer 和 RNG 状态。中断后用相同参数、同一个输出目录并加 `RESUME=1`，Trainer 会寻找最新的 `checkpoint-N`；上游会根据恢复步数重设数据采样种子，因此连续和中断训练的批次顺序可能不同。

## 在验证集上测量动作误差

评估传入完整的 `22d` 目录，以其中 `meta/split.json` 的 `val_episode_indices` 定位原始 episode。评估读取检查点的相机配置和训练统计量，选择同一帧的视觉、状态、任务文本，在同步的 16 帧绝对关节目标上计算误差。

```bash
python training/evaluate.py \
  --dataset data/processed/22d \
  --checkpoint outputs/rgb3_pir2/checkpoint-50000 \
  --out outputs/rgb3_pir2/validation.json
```

默认窗口从每个验证 episode 的第 0 帧开始，步长 16，尾部不足 16 帧的窗口跳过。每个窗口独立执行四步完整去噪；`--denoise`、`--stride`、`--seed` 可显式设置。报告包含每个 episode 的窗口数、按每个标量元素加权的全局 MAE/RMSE，以及四组关节分别的 MAE/RMSE。对于单位不同的手和臂，请分别读取组内误差。部署时的异步闭环行为需要用真机部署或回放入口检查。

使用 `--max-windows 1` 可快速检验完整读取和模型推理。没有验证集的冒烟样本可显式加 `--episode 0`。这时报告会将样本标记为显式选择。原 50k 包装脚本硬编码了 14 个验证集 ID，并依赖旧机器上缺失的评估实现；本仓库提供上述公开协议，便于在各自的数据切分上重跑。
