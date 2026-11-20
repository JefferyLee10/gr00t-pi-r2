# 复现范围与验证记录

## 固定版本

| 组件 | 源码版本 | 本项目怎样使用 |
|---|---|---|
| πR² 训练 fork | [Isaac-GR00T · 2a39d59](https://github.com/pi-r2-flow/Isaac-GR00T/tree/2a39d591a3af24c42bb1d1c9cd708a2dddac0600) | GR00T-N1.7 的 πR² 模型、数据加载器与微调入口 |
| πR² 推理 fork | [Isaac-GR00T · 9a00a47](https://github.com/pi-r2-flow/Isaac-GR00T/tree/9a00a4739c05b7638931f505fcc7a3c6f1502187) | 启动解耦模型服务，对外提供视觉特征更新、动作生成和流式状态管理 |
| πR² 客户端 | [pi-r2-flow · 3af52ca](https://github.com/pi-r2-flow/pi-r2-flow/tree/3af52ca400a6ec7d141416879aa531a6f62f697a) | 三相机和单头相机 22 维客户端沿用其请求协议 |
| 两个初始模型 | [完整修订号与文件说明](../weights/README.md) | GR00T 基础模型和 Cosmos-Reason2 视觉语言骨干 |

`scripts/setup_sources.sh` 按提交号拉取到 `third_party/`。训练端保留原项目的绝对动作统计修复。两个模型 checkout 都在加载 Cosmos 模型和 processor 时支持 `GR00T_COSMOS_LOCAL_PATH`，以便换机后使用自己的本地权重。上游 Isaac-GR00T 归属和许可见 [NOTICE](Isaac-GR00T-NOTICE) 与 [Apache License 2.0](Isaac-GR00T-LICENSE)。

## 哪套数据对应哪次实验

| 资料 | 已知范围 |
|---|---|
| 单头相机录制导出 | 1,293 个 episode、1,189,157 帧，其中 1,164 个训练 episode。原项目留有 317,000 步微调的 checkpoint 和双服务部署记录 |
| 三相机长时录制导出 | 1,315 个 episode、1,235,270 帧，其中 1,183 个训练 episode。保留了 17h 数据转换和训练视图 |
| 三相机 50k 配方 | 来自较早的三相机实验；其训练视图引用的旧数据目录已不可用。17h 数据兼容该配方。精确重现实验数值需要原始那份示教、切分与训练状态 |

源导出的元数据记录了 `rdf_to_lerobot_v4`、30→20 Hz 和 `absolute_state_t_plus_1`。项目目录中可以追溯的起点是平台导出的 **28 维 LeRobot v3.0**。使用现场原生 RDF 时，需要采集平台提供对应导出器。仓库包含训练与控制侧的软件，机器人上的 Pyro5 服务和硬件驱动由设备方提供，接口与只读回放替代件见[部署说明](../deployment/README.md)。

## 发布版实测

- 从远端获取三个固定提交，重复执行安装脚本成功；Linux x86_64 环境用上游锁文件安装了 Python 3.10、PyTorch 2.7.1+cu128 和 TorchCodec 0.4.0。
- 对两种真实源导出各转换一个 1,709 帧 episode。三相机样本三个阶段的 Parquet 数值逐项对齐历史产物，三路 MP4 逐字节一致。完整导出的全部 3,945 个三相机区间与 1,293 个头部相机区间均核对了源视频存在性和时长。全量重编码未在本次执行。
- 使用原三相机 50k checkpoint、独立空的 Hugging Face 缓存和本地 Cosmos 快照，在三相机长时数据切分的一个验证窗口上完成 16×22 离线预测。该窗口用于核对数据到模型的接线。
- 使用同一 checkpoint 在两块 RTX 5880 Ada GPU 上分别启动 VLM 和 DiT 服务。通过本机只读 Pyro5 桥回放真实录制帧，两秒干运行完成视觉缓存预热、连续动作块交换和 20 Hz 控制循环，正常退出。未连接实体机器人。
- 小型合成数据回归和控制器协议测试可按下列命令复跑。历史阈值变体可用 `bash deployment/run_variant.sh --list` 查看。

```bash
python data_pipeline/test_prepare.py
python deployment/test_contract.py
```

历史环境使用的 pilot 处理、逐帧审计、三相机与单头相机配置、离线训练启动器、评估包装器及各阶段的控制脚本，保存在 [lab-reference 分支](https://github.com/JefferyLee10/gr00t-pi-r2/tree/lab-reference)。其中的原始评估包装器引用了已不存在的外部实现。主分支提供可运行的独立窗口 MAE/RMSE 评估，详见[评估协议](../training/README.md)。
