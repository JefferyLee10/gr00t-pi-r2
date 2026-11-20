# 从 LeRobot v3.0 导出到 GR00T 训练输入

`prepare.sh` 把整段录制的视频与表格按 episode 切开，对左右臂和手部数据做 28→22 维投影，然后按已有切分生成训练集。三相机与单头相机配置共用同一套动作定义。

## 环境与命令

在仓库根目录执行。数据转换使用 CPU，需 Python 3.10、FFmpeg（启用 libx264）和 ffprobe。验证环境使用 FFmpeg 4.4.2。可单独建环境，与训练依赖隔离。

```bash
python3.10 -m venv .venv-data
source .venv-data/bin/activate
python -m pip install -r data_pipeline/requirements.txt
ffmpeg -hide_banner -encoders | grep libx264
ffprobe -version
bash data_pipeline/prepare.sh data/raw/my_dataset data/processed rgb3
```

`rgb3` 需要头部、左腕和右腕三路 RGB。单头 RGB 的完整运行命令为 `bash data_pipeline/prepare.sh data/raw/my_dataset data/processed headrgb`。测试两种配置时选用不同的输出目录。脚本接受绝对路径，`PYTHON` 环境变量可指定解释器。

先检查一个有训练样本的小前缀，可执行以下命令。正式处理时使用新的输出位置并去掉 `--limit-episodes`。

```bash
bash data_pipeline/prepare.sh data/raw/my_dataset outputs/data-smoke rgb3 --limit-episodes 2 --video-workers 2
```

## 输入合同

目录结构和关节顺序见[源数据说明](../data/README.md)。每个 episode 的数据须位于一个 Parquet 分片内，每路视频的对应片段须位于一个恒定帧率的视频文件内。源导出如果包含深度流，脚本只读取配置所列的 RGB 流。

| 文件 | 读取字段 |
|---|---|
| `meta/info.json` | `codebase_version = "v3.0"`、`fps`、`total_frames`、`total_episodes`、特征表 `features`，可选的 `chunks_size` 默认为 1000 |
| `meta/modality.json` | `state` 与 `action` 中 `left_arm/left_gripper/right_arm/right_gripper/left_hand/right_hand` 的 `start/end`，使用绝对关节量 |
| `meta/tasks.parquet` | 每行 `task_index`、`task`，编号从零连续，文本与采集任务匹配 |
| `meta/split.json` | `episodes` 数组覆盖所有 episode，每行含 `episode_index`、`episode_name`、`split`（`train` 或 `val`） |
| `meta/episodes/chunk-*/file-*.parquet` | 每行 `episode_index`、`length`、`dataset_from_index`、`dataset_to_index`、`data/chunk_index`、`data/file_index`；每个摄像头有 `videos/<完整特征名>/{chunk_index,file_index,from_timestamp,to_timestamp}` |
| `data/chunk-*/file-*.parquet` | 每帧的 `observation.state`、`action`（各 28 个 float）、`timestamp`、`episode_index`、`frame_index`、`index`、`task_index`。episode 和全局帧索引从零开始，单条 episode 内的帧索引从零重置 |
| `videos/<完整特征名>/chunk-*/file-*.mp4` | 特征名为 `observation.images.head_rgb` 等，视频时间单位为秒；片段长度乘以数据帧率应等于 episode 帧数 |

例如，两条示教的切分文件可以写为

```json
{
  "episodes": [
    {"episode_index": 0, "episode_name": "trial_a", "split": "train"},
    {"episode_index": 1, "episode_name": "trial_b", "split": "val"}
  ]
}
```

参考导出使用 20 Hz。其采集流程把下一时刻的绝对状态写入 `action`。转换器保留源端状态和动作的时间关系。放入自行导出的数据时，应先确认动作定义、单位以及左右手顺序。若要复现实验划分，使用原导出的 `split.json`。

## 输出与核验

| 输出目录 | 用途 |
|---|---|
| `28d/` | 保留全体选中 episode 的原始数值，按起始时间解码视频，用 H.264 重新编码，并以 ffprobe 核验每个片段的帧数 |
| `22d/` | 保留全体选中 episode、源编号和 `meta/split.json`。两臂各 7 维，每手 4 维。离线验证使用这里的 `val_episode_indices` |
| `train_view/` | 仅包含 `train`，重新连续编号，含 `meta/source_episode_mapping.jsonl` 和训练归一化文件 `meta/stats.json`。把此目录传给训练入口 |

各阶段的摄像头名称、动作分组和任务标签通过 `meta/modality.json` 提供。训练统计从最终训练 Parquet 逐帧汇总，每个浮点特征写出 `min/max/mean/std/q01/q99`。两个中间视图保留逐 episode 统计。为供验证加载器读取，`22d/meta/stats.json` 写入同一份训练统计。视频文件在三个输出目录之间使用硬链接，共享同一份编码内容。输出位于同一个支持硬链接的 Linux 文件系统。

历史三相机数据包含 1,315 个 episode，其中 1,183 个训练、132 个验证。历史单头相机数据包含 1,293 个 episode，其中 1,164 个训练、129 个验证。新数据的计数由其自身切分决定。

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("data/processed")
source = json.loads((root / "22d/meta/split.json").read_text())
train = root / "train_view/meta"
mapping = [json.loads(line) for line in (train / "source_episode_mapping.jsonl").read_text().splitlines()]
assert [row["source_episode_index"] for row in mapping] == source["train_episode_indices"]
assert set(source["train_episode_indices"]).isdisjoint(source["val_episode_indices"])
stats = json.loads((train / "stats.json").read_text())
assert all(len(stats[key]["q99"]) == 22 for key in ("observation.state", "action"))
print(json.loads((train / "info.json").read_text())["total_frames"], "training frames")
PY
```

在没有自有录制的机器上，运行 `python data_pipeline/test_prepare.py` 可在临时目录生成三条短示教。测试经过实际视频编码、切片和 Parquet 转换，并核验跨分片编号、手部投影、训练统计排除验证样本，以及三相机文件的对应关系。结束后临时目录自动清理。
