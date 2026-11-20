# 自备数据

把一次完整的 LeRobot v3.0 导出放到 `data/raw/my_dataset/`。示教数据需要保留逐帧机器人状态、动作目标、任务文本、时间信息、相机视频以及每个 episode 在视频和 Parquet 分片中的位置。从已有导出开始时，可以把该目录作为转换命令的输入，避免再复制一份。

```text
data/raw/my_dataset/
├── data/chunk-000/file-000.parquet
├── videos/observation.images.head_rgb/chunk-000/file-000.mp4
├── videos/observation.images.left_wrist_rgb/...    # 三相机配置
├── videos/observation.images.right_wrist_rgb/...   # 三相机配置
└── meta/
    ├── info.json
    ├── modality.json
    ├── split.json
    ├── tasks.parquet
    └── episodes/chunk-000/file-000.parquet
```

原始训练导出的元数据记录了如下处理方式：用录制平台的 `rdf_to_lerobot_v4` 将 30 Hz 的 RobotDataset 1.0 录制转到 20 Hz，最短 episode 设为 5 秒，使用下一时刻的绝对状态作为当前时刻的动作目标（`absolute_state_t_plus_1`）。每个任务应沿用自己的动作定义和任务文本。

一个分片可以包含多个 episode。源导出如含 `stats.json` 和深度流，保留它们即可。输入数据的 28 维顺序如下，两臂关节按弧度表示，状态与动作使用相同分组。

| 源切片（Python 左闭右开） | 输入 | 目标 |
|---|---|---|
| `0:7`、`8:15` | 左臂、右臂各七个绝对关节值 | 依次写入目标 `0:7`、`7:14` |
| `7`、`15` | 左右夹爪各一维 | 在投影时去除 |
| `16:22`、`22:28` | 左右手各六个值，按各手设备的训练顺序排列 | 每手取前三个值，后三个求均值，写入目标 `14:18`、`18:22` |

如果原始素材还是录制设备导出的 SQLite/RDF、ROS bag 或分散的相机文件，先用采集系统导出上述结构并确认关节含义、单位、帧率和时间对齐。本项目的数据处理入口是 28 维 LeRobot 导出。只有 26 维、不同关节顺序或缺少动作目标的导出都需要按设备协议先做映射。完整转换步骤、切分和统计量见[数据流水线](../data_pipeline/README.md)。
