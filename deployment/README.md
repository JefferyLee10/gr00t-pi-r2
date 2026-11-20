# 双臂双手真机推理

两个控制器共用一个机器人接口。每帧读取双臂各 7 维、双手各 6 维的绝对关节角，单位为弧度。模型看到头部 RGB 或头部加双腕 RGB，以及按左臂、右臂、左手、右手排列的 22 维状态。手部压缩保留前三个关节，后三个取均值。输出是同顺序的 22 维绝对关节目标，客户端将其转换成实际的 26 维指令。

## 安装

从仓库根目录运行。GPU 机器先按[训练环境说明](../training/README.md)安装 GR00T 依赖，并准备[权重](../weights/README.md)。如果控制器单独运行在一台 CPU 机器上，可用下列命令单独创建并激活 Python 3.10 客户端环境。普通 GR00T 控制器会直接载入模型，需要在 GPU 环境中运行。

```bash
bash scripts/setup_sources.sh
# 单独的 CPU 客户端执行下面两行。GPU 端沿用已激活的训练环境。
uv venv --python 3.10 .venv-client
source .venv-client/bin/activate

uv pip install --python "$VIRTUAL_ENV/bin/python" -r deployment/requirements.txt
python deployment/test_contract.py
```

其中最后一条命令在 CPU 上检查合成 JPEG、两种相机布局、关节顺序、手部压缩与动作限幅，全程不连接机器人。

两个模型服务各载入一份完整 checkpoint。两张 GPU 可以位于不同机器上，各机器都需要同一份 checkpoint、backbone 和部署源码。每个控制器使用专属的一对服务，服务进程保存上一次请求的推理状态。

## 示例 IP 与端口

| 地址 | 原项目用途 | 换到自己的设备时 |
| --- | --- | --- |
| `10.0.0.11:7788` | 局域网内的 Pyro5 机器人桥，提供传感器数据并接收绝对关节目标 | 将 `--robot-host` 和 `--robot-port` 改为自己的机器人桥地址 |
| `127.0.0.1:5555` | 控制器连接的 VLM 服务 | 同机保留回环地址，跨机器使用自己的 VLM 主机地址或 SSH 转发端口 |
| `127.0.0.1:5556` | 控制器连接的 DiT 服务 | 同机保留回环地址，跨机器使用自己的 DiT 主机地址或 SSH 转发端口 |
| `0.0.0.0` | 原启动脚本的服务端监听地址，用于所有网卡 | 本仓库默认监听回环；跨机器直连时将 `serve.sh` 的第四个参数换成自己的 GPU 机局域网 IP |
| `192.168.1.244` | 上游 xArm6 示例的单臂 SDK 地址 | 仅当使用上游 xArm6 单臂示例时替换其 `--arm-ip`，本项目双臂控制器使用上表的机器人桥 |

## πR² 双服务推理

在两个终端中分别启动一个服务并等待各自输出 `Server ready`。以下用 GPU 0 和 GPU 1，按机器的可用显卡编号修改。历史启动脚本使用 GPU 0 和 GPU 2，也有使用 GPU 2 和 GPU 3 的运行记录。用同一个目录替换两个命令中的 checkpoint 示例路径。

```bash
# 终端一，VLM
bash deployment/serve.sh weights/checkpoints/my-run 0 5555
```

```bash
# 终端二，DiT
bash deployment/serve.sh weights/checkpoints/my-run 1 5556
```

第三个终端负责控制。用训练时的相机配置和任务文本，下面的例子对应三路 RGB 和“快递分拣”。头部单相机 checkpoint 使用 `--cameras head`。两个控制器都要求显式提供机器人地址。

```bash
python deployment/pir2.py --robot-host 10.0.0.11 --robot-port 7788 \
  --vlm-host 127.0.0.1 --vlm-port 5555 --dit-host 127.0.0.1 --dit-port 5556 \
  --cameras rgb3 --task "快递分拣" --seconds 20
```

默认干运行会读取观测并完成真实推理，不申请控制权。检查日志中的 `warm-start = OK`、`VLM cache prime = OK`、动作范围与图像延迟。准备好实机执行时，在同一命令末尾加 `--send`。正常退出或 Ctrl+C 时客户端释放本次会话的控制租约，再用 Ctrl+C 分别停止两个服务。执行前请核对桥接层的关节顺序、角度单位、指令超时及急停功能与设备一致。

控制频率 20 Hz，默认 `--nfe 24 --initial-d 2`。运行时按 DiT 查询延迟估计每次输出的步数 d，训练支持范围为 1 至 5；主循环从预测序列的 `[d, 2d)` 段发出动作。收到新观测后异步更新 VLM 特征。对于 RGB3 V2，任一手臂原始目标相对测量值偏离超过 70°、相邻预测偏离超过 15°，或者当前观测超过 150 ms 均可触发拒绝。头部单相机 V1 相邻预测门限为 8°。手臂每次最多变化 15°，物理手关节每次最多变化 1 rad。手部除首关节外按一维开合量驱动，系数为 0.6、0.8、0.8、0.8、0.8。以上是该机器人历史配置值，换机器人时需要重新评估。

## 无机器人时复现控制链

GPU 服务照常启动。另开一个终端运行只读回放桥，指定数据转换第一阶段生成的 **28 维 v2.1 数据**。脚本解码指定帧，并用新的时间戳重复提供这一组关节角和图片。

```bash
python scripts/replay_bridge.py --dataset data/processed/28d --episode 0 --frame 0 --port 17788
```

```bash
python deployment/pir2.py --cameras rgb3 --robot-host 127.0.0.1 --robot-port 17788 --seconds 10
```

控制器、VLM/DiT RPC、视觉编码、动作解码和限幅都经过同一条代码路径。固定帧没有环境反馈，可用于检查接线和模型兼容性。要衡量预测相对真实动作的误差，请使用[离线评估](../training/README.md)。

## 普通 GR00T 基线

使用普通 flow 训练得到的 22 维 checkpoint，同步在一个进程内运行模型，取每个动作块的第一个目标。默认 4 次去噪，手臂原始目标门限 15°，每次手臂与手部变化上限分别为 2° 和 0.05 rad。该基线的手部展开是直接重复第四个关节值。可用相同回放桥或真实机器人桥。

```bash
python deployment/gr00t.py --checkpoint weights/checkpoints/my-plain-flow \
  --robot-host 10.0.0.11 --robot-port 7788 --cameras rgb3 --task "快递分拣" --seconds 20
```

追加 `--send` 才会下发命令。两种控制器的 `--seconds 0` 均表示运行到手动停止。

## 机器人桥接口

真实硬件依赖机器人端已有的 `model_inference_bridge` 服务及其底层驱动。本机工程中保存了客户端和下表的接口，硬件端服务源码需要由机器人提供方提供或适配。Pyro5 对象 URI 为 `PYRO:model_inference_bridge@<自己的机器人IP>:<端口>`，实验使用 Pyro5 5.17、自定义 pickle 序列化器、协议 ID 5。在受信任的设备网络内运行该接口；读写双方必须使用相同的线协议。

| 调用 | 输入或返回值 |
| --- | --- |
| `ping()` | 原桥返回 `pong`，只读桥返回真值 |
| `get_obs_status()` | 例如 `{"ready": True, "missing": []}` |
| `get_observation()` | 暂未就绪可返回 `None`。就绪时返回字典，有递增的整数 `timestamp_ns`、关节数组 `left_arm[7]`、`right_arm[7]`、`left_hand[6]`、`right_hand[6]`，及 JPEG 字节串 `head_rgb_jpeg`。三相机增加 `left_wrist_rgb_jpeg` 和 `right_wrist_rgb_jpeg` |
| `submit_command(command)` | 字典有字符串 `client_uuid` 和上述四组关节的浮点列表，单位 rad。发送本周期的最新绝对目标 |
| `release_control_lease(client_uuid)` | 使用提交命令时的同一个 UUID，释放控制权 |

## 历史 RGB3 控制实验

`pir2.py` 对应三相机 V2 或单相机 V1。其余十个实验以可执行的补丁保存在 `variants/`，用系统的 GNU patch 在临时目录生成对应控制器。名称对应原始脚本后缀，包括初版、速度缩放、拒绝次数、抓取同步、接触及抓后保持。各实验沿用自己的阈值和手部逻辑，可以先查看专属参数再运行。

```bash
bash deployment/run_variant.sh --list
bash deployment/run_variant.sh v3_grasp_sync --help
bash deployment/run_variant.sh v2_slowarm7p5 --robot-host 127.0.0.1 --robot-port 17788 --seconds 10
```
