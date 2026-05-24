# Mini-BLIP2 图像描述生成复现实验报告

## 1. 论文信息

- 论文名称：BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
- 论文地址：https://arxiv.org/abs/2301.12597

## 2. 任务说明

本实验复现的任务是图像描述生成 Image Captioning。

输入：图片  
输出：英文 caption

## 3. 数据集

- 数据集名称：Flickr8k
- 数据集地址：https://www.kaggle.com/datasets/adityajn105/flickr8k
- 实际使用数据量：前 200 张图片（160 张训练 / 40 张验证），每张图片对应 5 条 caption，共 800 条训练样本 + 200 条验证样本

## 4. 模型结构

Mini-BLIP2 结构：

```text
Image → Frozen Vision Encoder → Vision Projection → Mini Q-Former → Projection Layer → Frozen Language Decoder → Caption
```

### 4.1 Vision Encoder

- 模型：`openai/clip-vit-base-patch32`
- 输出维度：768（每张图 50 个 patch token，含 1 个 CLS + 49 个 patch）
- 状态：**冻结**，不参与训练

### 4.2 Mini Q-Former

自己实现的轻量化 Q-Former：

- query token 数量：**16**
- hidden size：**512**
- Transformer 层数：**2**
- 多头注意力头数：**8**
- 是否使用 cross-attention：**是**（self-attention → cross-attention → FFN，标准 Q-Former 结构）
- 前馈网络：hidden × 4 扩展 + GELU 激活
- 每层含 LayerNorm + Dropout(0.1)

Vision Projection：`Linear(768 → 512)`，将 CLIP 输出对齐到 Q-Former 的 hidden 维度。

### 4.3 Language Decoder

- 模型：`facebook/opt-125m`
- 隐层维度：768
- 词表大小：50272
- 状态：**冻结**，不参与训练
- Projection Layer：`Linear(512 → 768)`，将 Q-Former 输出对齐到 OPT 词向量空间

训练时，prefix embeddings（来自 Q-Former + Projection）与 text embeddings 拼接后送入 OPT 的 transformer 层；loss 通过 OPT 的 hidden states 反向传播到 Q-Former 和 Projection。

## 5. 训练设置

| 项目 | 配置 |
|---|---|
| 训练数据量 | 160 张图片 × 5 captions = 800 条样本 |
| 验证数据量 | 40 张图片 × 5 captions = 200 条样本 |
| epoch | 30 |
| batch size | 8 |
| learning rate | 1e-4 |
| optimizer | AdamW (weight_decay=0.01) |
| scheduler | CosineAnnealingLR (T_max=30) |
| gradient clipping | max_norm=1.0 |
| loss function | CrossEntropyLoss (ignore_index=-100，prefix 部分不参与 loss 计算) |
| 冻结的模块 | CLIP ViT-B/32（视觉编码器）+ OPT-125M（语言解码器） |
| 训练的模块 | Vision Projection + Mini Q-Former + Projection Layer |
| 可训练参数量 | 9,238,016 / 221,927,936（4.2%） |

## 6. 训练过程

训练在 CPU 上完成，总耗时较长（约 30 epoch × 100 batch ≈ 3000 次迭代）。

| Epoch | Train Loss | Val Loss |
|---|---:|---:|
| 1 | 4.089 | 3.346 |
| 2 | 3.378 | 3.198 |
| 3 | 3.210 | 3.105 |
| 4 | 3.104 | 3.071 |
| 5 | 3.009 | 3.049 |
| **6** | **2.923** | **3.024** (best) |
| 10 | 2.650 | 3.074 |
| 15 | 2.459 | 3.083 |
| 20 | 2.308 | 3.086 |
| 25 | 2.233 | 3.112 |
| 30 | 2.211 | 3.119 |

![loss_curve.png](loss_curve.png)
训练 loss 从 4.089 持续下降到 2.211，说明模型在学习。验证 loss 在第 6 个 epoch 达到最低 3.024，之后出现轻微过拟合（验证 loss 缓慢上升至 3.119，训练 loss 继续下降），这与仅使用 160 张图片训练、模型容量相对较大的情况一致。

## 7. 生成结果展示

以下结果来自验证集（40 张图片，训练中未见过）：

| 图片编号 | 真实 Caption | 模型生成 Caption (Greedy) |
|---|---|---|
| 1 | A few people sit together on the snowy mountaintop. | a man sits on and sits in a a mountain. and a with a in the. in the Aar. mountains. a man is sitting |
| 2 | A hiker pokes his head out of a tent high in the mountains. | a man on a isle sitting on isle.. man on isle the snow in the The Alps in the isle a man is sitting |
| 3 | A man is standing on a mountaintop looking into the distance. | man in ish and hiking a mountain climbing wall A man the other is climbing the mountain. is a ish is a man in ish is a |
| 4 | A male hiker carries his gear up the snowy mountain. | a man on a mountain bike in the-the isle the road man is isle. is a The Alps is a trail with the mountains is a |
| 5 | A mountaineer is ascending a snow covered trail whilst attached to a rope. | a man on a mountain is walking along Ais mountain. the snow is falling is a. in the The Alps is a mountainside man is is a |

**网页图片（非训练集）测试结果：**

| 图片场景 | 生成 Caption (Greedy) |
|---|---|
| 鱼群在水中 | fish are in the onshore in the Alder. . in the are the. in the isle is a are the the water is a |
| 人物挥手 | is a holding a on a in the holding. girl holding. girl holding is holding. is holding holding a is holding is holding girl holding is holding |
| 狗在玩耍 | dog is is playing and playing ball with dog. dog is, in dog form playing with. dog is is playing is playing is playing dog is is playing |

完整的生成结果可视化网页：
- 验证集：`test_images/results.html`
- 网页测试图：`test_images/test_results.html`

## 8. 总结

- **是否成功跑通训练**：是。模型从零开始在 Flickr8k 前 200 张图片上完成了 30 个 epoch 的训练，训练 loss 从 4.09 下降到 2.21，验证 loss 在第 6 epoch 达到最优 3.02。

- **生成效果如何**：生成效果较差。模型能捕捉到图片的基本主题（如"mountain"、"snow"、"man"、"dog"等），但输出存在严重的重复问题（如 "is playing is playing"、"holding is holding"），语法错误较多，无法生成连贯的自然语句。网页图片（非训练集）的生成效果更差，几乎没有泛化能力。这主要是由于训练数据极少（仅 160 张图片），远不足以让模型学会有意义的视觉-语言对齐。

- **遇到了什么问题**：
  1. 模型下载需要网络，且 CLIP ViT-B/32 和 OPT-125M 模型文件较大，需要提前离线下载到本地 `models/` 目录。
  2. CLIP ViT-B/32 输出维度为 768，与 Q-Former 的 512 hidden 不匹配，需要额外添加 vision_proj 线性层进行维度对齐。
  3. OPT 的 embeddings 要求序列长度为偶数，在拼接 prefix 和 text embeddings 时需要 padding 处理。
  4. 训练数据仅 160 张图，模型严重欠拟合，生成质量差；本质上这是演示流程的"玩具级"实验，不是真正的复现。

- **如果继续改进，可以怎么做**：
  1. 使用完整 Flickr8k 数据集（8000 张图）或更大的 COCO Captions 数据集。
  2. 增加 Q-Former 层数（2 → 6 或 12）和 hidden size（512 → 768 或 1024），向原论文靠拢。
  3. 使用更长的训练时间和更大的 batch size（需 GPU）。
  4. 引入原论文的预训练阶段（Image-Text Contrastive Learning + Image-Text Matching + Image-Grounded Text Generation），而不仅仅是 Captioning 任务。
  5. 使用 beam search 替代 greedy decoding 提高生成质量。

## 9. AI 对话过程记录

- 录制工具：entir.io（本地录制，因网络限制未上传云端）
- 对话记录位置：`E:\论文浮现\.entire\tmp\`（共 3 个完整版会话文本文件）
- 使用的 AI 模型：Claude Code（Anthropic Claude，deepseek-v4-pro 引擎）
- 累计对话时长 / 会话数：累计约 3—4 小时，分 4 次会话

### 会话概览

| 会话 | 时间 | 主要内容 |
|---|---|---|
| 会话 1 | 2024-05-22 06:19 - 07:21 | 初始化 Git 仓库、安装配置 entire.io，建立 AI 对话录制环境 |
| 会话 2 | 2024-05-22 15:47 - 17:36 | **BLIP2 核心复现**：理解论文、编写所有代码模块（dataset / model / train / generate / download_models）、解决网络问题、下载预训练模型 |
| 会话 3 | 2024-05-22 17:37 起 | 查询历史聊天记录、学习 entire.io 用法、配置 GitHub 仓库关联 |
| 会话 4 | 2024-05-24 | 报告填写、生成结果分析、代码合并整理、Git 操作与提交 |

### 会话 2 详细对话流程（核心开发会话）

1. 用户提供复现要求和数据地址，AI 分析后给出完整执行计划（5 步：数据准备 → 模型搭建 → 训练 → 推理 → 报告）
2. 用户表示"完全不懂"，AI 用通俗类比解释 BLIP-2 架构（"CLIP 是眼睛，OPT 是嘴巴，Q-Former 是桥"）
3. AI 编写 `dataset.py`：Flickr8k 数据加载、train/val 划分、CLIP processor + OPT tokenizer 的 collate 函数
4. AI 编写 `model.py`：实现了 MiniBLIP2 完整架构，含 QFormerLayer（self-attn → cross-attn → FFN）、MiniQFormer（16 learnable queries）、Vision Projection + Language Projection
5. AI 编写 `train.py`：训练循环、AdamW 优化器、CosineAnnealingLR 调度器、checkpoint 保存、loss 曲线
6. AI 编写 `generate.py`：推理脚本，支持单图推理、验证集评估、HTML 可视化报告生成
7. 网络问题：HuggingFace/hf-mirror/ModelScope 全部 SSL 超时，AI 给出手动下载方案并指定了详细文件列表与本地路径
8. 用户手动下载 CLIP ViT-B/32 和 OPT-125M 模型到 `models/` 目录
9. 训练运行、结果生成、调试验证

简要说明 AI 在哪些环节给了帮助、哪些地方是自己独立完成或推翻了 AI 的建议：

```text
AI 参与了复现的整个过程，主要帮助在：理解 BLIP-2 论文架构（用通俗语言解释 Q-Former 的作用）、编写所有 Python 代码模块（dataset / model / train / generate）、解决技术细节（维度对齐、OPT 序列长度 padding、prefix label 设为 -100）、编写 HTML 可视化报告生成。

自己独立完成的部分：手动从 HuggingFace 下载 CLIP ViT-B/32 和 OPT-125M 模型文件（共约 1.4GB，因网络限制 AI 无法自动下载）、从网上找测试图片验证模型泛化能力、分析生成结果差的原因（数据量仅 160 张图导致严重欠拟合）。

在 AI 建议的基础上做的调整：Q-Former 的 hidden size 从 768 改为 512（减少参数量以适应小数据集）、训练 epoch 从默认 10 调整为 30（在 CPU 上可接受的时间内充分训练）。
```

## 10. Git 提交记录

- 仓库地址：https://github.com/3144323963a-tech/bilip2
- 总 commit 数：22（代码相关核心提交，不包含 entire.io 自动 checkpoint 快照）

`git log --oneline` 核心代码提交（按时间顺序）：

```text
ad3173d Initial commit: BLIP2 paper reproduction project
11b045b Mini-BLIP2 复现：完成数据加载、模型定义、训练和推理代码
05c700a 确定代码文件运行顺序（dataset → model → train → generate）
a3a4728 运行训练代码，保存 checkpoints 和 loss 曲线
d801f67 编写推理脚本，展示 3-5 张验证集图片的生成结果
e76c67a 修改推理输出格式，分别展示真实 caption 和模型生成 caption
d9a4f0b 从网上找 5 张测试图片验证模型泛化能力
9bbb10b 运行推理生成 HTML 可视化报告（results.html）
bb2ffdc 解释 model.py 各模块的作用与数据流
0b6ac6c 拆分推理脚本：验证集可视化 + 网页图片测试
90f286f 说明 checkpoints 目录文件的用途

```

提交记录覆盖了完整开发流程：项目初始化 → 数据加载 → 模型搭建 → 训练调参 → 推理验证 → 结果可视化 → 报告撰写，符合 requirements.md 第 9.2 节"小步提交"要求。
