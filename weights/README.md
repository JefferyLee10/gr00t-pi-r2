# 权重目录

以下命令在仓库根目录运行。训练使用 GR00T 初始参数和 Cosmos 的视觉语言模型配置及处理器。推理使用针对本机器人微调后的 checkpoint。权重和训练状态都留在本地。

| 目录 | 放入什么 | 何时使用 |
|---|---|---|
| `weights/base/` | `nvidia/GR00T-N1.7-3B` 的完整 Hugging Face 快照，`config.json` 直接位于此目录 | 从基础模型微调 |
| `weights/backbone/` | `nvidia/Cosmos-Reason2-2B` 的完整快照，包含 tokenizer、processor 和配置 | 构造视觉语言模型、预处理器及离线运行 |
| `weights/checkpoints/<name>/` | 自己训练或取得的完整 GR00T checkpoint 目录，保留所有 JSON 和模型分片 | 离线评估、模型服务、真机推理 |

## 获取同版本初始模型

先进入训练用的 Python 环境。有权访问模型的 Hugging Face 账号可用 `hf auth login` 登入。以下脚本固定到本项目开发环境使用过的修订版本，写入上表的目录。手动下载时使用相同版本，或从已有缓存复制整个快照。Hugging Face 缓存的 snapshots 目录通常包含相对软链接，复制时需解引用，例如 `cp -aL /your/cache/snapshots/<revision>/. weights/base/`。

```bash
python scripts/prepare_weights.py download base
python scripts/prepare_weights.py download backbone
```

| 模型 | Hugging Face 修订版本 |
|---|---|
| GR00T-N1.7-3B | [2fc962b973bccdd5d8ce4f67cc63b264d6886495](https://huggingface.co/nvidia/GR00T-N1.7-3B/tree/2fc962b973bccdd5d8ce4f67cc63b264d6886495) |
| Cosmos-Reason2-2B | [9ce19a195e423419c349abfc86fd07178b230561](https://huggingface.co/nvidia/Cosmos-Reason2-2B/tree/9ce19a195e423419c349abfc86fd07178b230561) |

## 检查手动放入的权重

```bash
python scripts/prepare_weights.py check base
python scripts/prepare_weights.py check backbone
python scripts/prepare_weights.py check checkpoint --path weights/checkpoints/my-run
```

检查工具读取模型和处理器的 JSON、权重索引、各 safetensors 文件头与长度，可以发现缺失分片、截断文件和只下载了 Git LFS 指针的情况。模型内容是否适配本机器人，还要用对应相机配置做离线推理。

推理 checkpoint 至少保留 `config.json`、`processor_config.json`、`statistics.json`、`embodiment_id.json`，以及 `model.safetensors` 或索引和它列出的所有分片。复制时最稳妥的做法是保留整个 `checkpoint-<step>/`。如需从中断处恢复训练，还要保留训练器状态、优化器分片、学习率调度器和随机数状态。记录 checkpoint 的相机配置、任务文本和动作定义，使部署端与训练保持一致。
