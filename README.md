# 实验脚本存档

这个分支保留实验过程中使用过的项目脚本，用来追溯数据处理、训练配方及真机控制逻辑的演变。新的安装、训练和部署从 [main 的中文指南](https://github.com/JefferyLee10/gr00t-pi-r2/tree/main)开始。

| 位置 | 可以查到什么 |
|---|---|
| `scripts/`、根目录 `build_*`、`repair_*`、`audit_*` | v3.0→v2.1 的 FFmpeg 兼容处理、16h 小规模试跑、逐帧审核、28→22 维映射、训练视图与 17h 配置 |
| `experiments/*/`、`train_*.sh` | 实验运行时使用的模态注册文件、在线与离线启动参数、单头相机和三相机配方 |
| `deploy_*.py`、`run_pir2_headrgb_checkpoint_real.sh` | 原双臂客户端和试验参数，包括抓取时序、速度限制和历史实机启动方式 |
| `eval_one_pir2_checkpoint.py` | 原包装器、固定验证集和它依赖的外部评估脚本名称 |
| `learning/Isaac-GR00T/`、`absolute-actions.patch` | 当时的离线启动器及绝对动作统计量修复 |

这些是对照用的历史版本。它们引用实验时的系统路径、特定 episode 个数和设备编号，部分控制器直接发送真机命令。在自己的机器人上运行，请使用 main 中显式选择地址、默认干运行的版本。原脚本中的用户主目录统一替换为 `/path/to/lab-home`。根目录的 `ORIGINAL_SHA256SUMS` 记录替换前各本机源码的散列，可在原项目目录中核对。实验产生的录像、数值记录、日志、模型、优化器状态和缓存均由使用者在本地管理。
