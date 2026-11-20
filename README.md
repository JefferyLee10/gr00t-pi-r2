# GR00T + πR² · 双臂双手真机策略

用自己的机器人示教训练 GR00T-N1.7，并通过 πR² 的异步视觉和动作推理控制双臂双手。仓库提供从 **LeRobot v3.0 源数据**到训练、离线评估、只读回放和真机控制的完整工作流，支持头部单相机和头部加双腕三相机。原实验任务指令为「快递分拣」。

| 环节 | 输入 → 输出 | 使用说明 |
|---|---|---|
| 数据处理 | 28 维源导出 → 按 episode 对齐的 RGB、22 维绝对关节目标、独立训练集与统计量 | [输入格式](data/README.md) · [转换命令与预期结果](data_pipeline/README.md) |
| 微调 | GR00T、Cosmos 和训练集 → 自己的 checkpoint | [权重放置与校验](weights/README.md) · [训练、续训与离线评估](training/README.md) |
| 推理部署 | checkpoint、图像、关节状态 → 16 步动作块 → 20 Hz 控制 | [模型服务、IP 配置、回放和真机命令](deployment/README.md) |
| 实验追溯 | 原始配方、试验控制器、数据审计源码和对应版本 | [验证记录和版本](docs/reproduction.md) · [lab-reference 分支](https://github.com/JefferyLee10/gr00t-pi-r2/tree/lab-reference) |

## 从空仓库运行

下面以 Linux x86_64、CUDA 12.8、Python 3.10 和三相机为例。需要 Git、uv、带 libx264 的 FFmpeg 和 ffprobe，以及支持 BF16 的 NVIDIA GPU。正式训练和默认的双服务部署使用两张 GPU。数据转换和接口测试可单独在 CPU 上运行。

```bash
git clone https://github.com/JefferyLee10/gr00t-pi-r2.git
cd gr00t-pi-r2
bash scripts/setup_sources.sh
(cd third_party/groot_train && uv sync --frozen --python 3.10)
source third_party/groot_train/.venv/bin/activate
uv pip install --python "$VIRTUAL_ENV/bin/python" -r deployment/requirements.txt
```

**1. 放入数据和权重。** 将一份完整数据集放到 `data/raw/my_dataset/`，使 `data/raw/my_dataset/meta/info.json` 可以直接读取。历史源导出的状态和动作均为 28 维，动作记录的是下一时刻的绝对状态。若只有原生 RDF/SQLite 或 ROS bag，请先从采集系统导出[指定的数据格式](data/README.md)。

```bash
python scripts/prepare_weights.py download base
python scripts/prepare_weights.py download backbone
```

手动准备时，两个完整模型快照分别放到 `weights/base/` 和 `weights/backbone/`。如果已有适配此机器人的微调权重，可将整个 checkpoint 放到 `weights/checkpoints/my-run/`，直接执行第 3 步的评估或第 4 步的部署。空占位目录已随 Git 提交，其内容由各使用者在本地管理。

**2. 转换数据。** 首次使用可以先转换一小段，检查相机、episode 和时间轴，再处理全量。

```bash
bash data_pipeline/prepare.sh data/raw/my_dataset outputs/data-smoke rgb3 --limit-episodes 2
bash data_pipeline/prepare.sh data/raw/my_dataset data/processed rgb3
```

最终的 `data/processed/train_view/` 只包含训练示教。`data/processed/22d/` 包含完整切分，供验证集评估。`data/processed/28d/` 保留两臂和两手的原始维数，可用来回放机器人的观测。

**3. 训练并评估。** 下面两条命令分别启动三相机 50k 配方和检查一个验证窗口。完整验证时去掉 `--max-windows 1`。头部单相机使用 `headrgb` 做转换和训练，原配方为 317k 步。在自己的数据上完整运行前，可按[训练指南](training/README.md)缩小批量和训练步数。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash training/train.sh rgb3 pir2
CKPT=outputs/rgb3_pir2/checkpoint-50000
python training/evaluate.py --checkpoint "$CKPT" --dataset data/processed/22d \
    --max-windows 1 --out outputs/validation.json
```

**4. 启动 πR² 服务和控制器。** 每个终端都从仓库根目录激活上面的运行环境。前两个终端使用同一个微调 checkpoint。下面第 3 个终端使用录制帧启动回放桥，第 4 个终端运行控制器，两者均可在没有机器人的情况下联调。

```bash
# 终端 1
bash deployment/serve.sh outputs/rgb3_pir2/checkpoint-50000 0 5555
# 终端 2
bash deployment/serve.sh outputs/rgb3_pir2/checkpoint-50000 1 5556
# 终端 3
python scripts/replay_bridge.py --dataset data/processed/28d --episode 0 --frame 0 --port 17788
# 终端 4
python deployment/pir2.py --cameras rgb3 --robot-host 127.0.0.1 --robot-port 17788 \
    --vlm-host 127.0.0.1 --vlm-port 5555 --dit-host 127.0.0.1 --dit-port 5556 \
    --task "快递分拣" --seconds 10
```

**换到真机时，请用自己的 IP 和端口。** 原实验机器人桥为 `10.0.0.11:7788`，替换控制器的 `--robot-host` 和 `--robot-port`。两个 `127.0.0.1` 指向控制器所在主机；若推理服务分布在其他机器，请分别设置 VLM、DiT 的服务端监听地址和客户端连接地址。控制器默认只读取观测并推理，确认关节顺序、单位和限幅后加 `--send` 才会下发动作。机器人端需有兼容的 Pyro5 桥和底层驱动。单头相机的部署参数是 `--cameras head`。完整参数、桥接口和标准 GR00T 基线在[部署指南](deployment/README.md)。

## 上游项目

πR² 的模型、流式推理协议和客户端来自 Sungjae Park 与 Shubham Tulsiani 的 *πR²: Reactive Real-time Flow Policies*。本仓库提供双臂双手的示教转换、模态配置、实验配方和控制器。固定版本与依赖来源见[版本说明](docs/reproduction.md)。
