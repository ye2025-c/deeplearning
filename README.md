# 深度学习自学仓库

> 个人深度学习学习记录：理论笔记 + 代码练手 + 项目实现，按阶段分层管理。
>
> **当前阶段**：阶段 1（预备知识 + 线性模型）& 阶段 5（Transformer 项目并行）→ 详见 [PROGRESS.md](PROGRESS.md)

---

## 目录结构

```
deeplearning/
├── README.md                   # 总路线图（本文件）
├── PROGRESS.md                 # 逐章勾选进度表
│
├── deeplearning notebook/      # 个人 Markdown 笔记（按章节命名）
│   ├── 深度学习初步.md
│   ├── 线性神经网络.md
│   ├── 查阅文档.md
│   ├── 分离计算的作用.md
│   ├── 线性回归实例分析.md
│   ├── 线性回归掌握计划.md
│   └── 线性回归简洁实现笔记.md
│
├── learning_process/           # 独立 .py 小实验（验证概念用）
│   ├── calculus.py
│   ├── 正态分布.py
│   ├── 矢量加速.py
│   ├── Timer.py
│   ├── 线性回归实例.py
│   └── 线性回归简洁实现.py
│
├── transformer/                # 阶段 5：从零实现 Transformer
│   └── transformer.py
│
└── (待建)
    ├── pytorch/                # D2L 教材配套 notebook（跟练用）
    ├── projects/               # 独立课题（MNIST / 情感分类 / 微调 BERT 等）
    └── data/                   # 自建数据集（D2L 数据由 d2l 自动下载）
```

---

## 学习路线（五阶段）

### 总览

| 阶段 | 主题 | 预计时长 | 状态 |
|------|------|----------|------|
| 0 | 环境与 Python 基础 | 1 周 | ✅ 完成 |
| 1 | 预备知识 + 线性模型 + 优化 | 2~3 周 | 🔄 进行中 |
| 2 | 多层感知机与正则化 | 2 周 | ⬜ 待开始 |
| 3 | 卷积神经网络与计算机视觉 | 3~4 周 | ⬜ 待开始 |
| 4 | 序列模型、注意力与 NLP | 3~4 周 | ⬜ 待开始 |
| 5 | 从零实现大模型（Transformer） | 持续推进 | 🔄 训练已跑通 |

---

### 阶段 0 · 环境与 Python（✅ 已完成）

**过关标准**：能 `import torch` 运行脚本，会看报错。

- [x] Python 3.10+、PyTorch、d2l 安装完成
- [x] 第一个 tensor 实验跑通

---

### 阶段 1 · 预备知识 + 线性模型（🔄 进行中）

**过关标准**：不看答案能独立写出「合成数据 → DataLoader → 训练循环 → 打印 loss」。

#### 预备知识 `pytorch/chapter_preliminaries/`

| 任务 | 配套练手 | 笔记 |
|------|----------|------|
| 张量操作 `ndarray.ipynb` | — | ✓ |
| pandas 数据处理 | — | ✓ |
| 线性代数 | — | `深度学习初步.md` ✓|
| 微积分 `calculus.ipynb` | `learning_process/calculus.py` ✓ | `深度学习初步.md` ✓ |
| 概率 `probability.ipynb` | `learning_process/正态分布.py` ✓ | — |
| 自动微分 `autograd.ipynb` | — | `深度学习初步.md` ✓ |

#### 线性模型 `pytorch/chapter_linear-networks/`

| 任务 | 配套练手 | 笔记 |
|------|----------|------|
| 线性回归原理 | `learning_process/线性回归实例.py` ✓ | `线性神经网络.md` ✓ |
| 线性回归从零实现 | — | ✓ |
| 线性回归简洁实现 | — | ✓ |
| Softmax 回归 | 新建 `learning_process/softmax_scratch.py` | 补充 `线性神经网络.md` |

#### 优化基础 `pytorch/chapter_optimization/`

| 任务 | 建议 |
|------|------|
| SGD / 动量 / Adam | 实验：同数据集换优化器对比 loss 曲线 |
| 新建笔记 `优化算法.md` | 记录超参影响规律 |

---

### 阶段 2 · 多层感知机与正则化（⬜ 待开始）

**过关标准**：能解释过拟合原因，会用 Dropout / 权重衰减控制验证 loss。

| D2L 章节 | 练手目标 |
|----------|----------|
| `chapter_multilayer-perceptrons/` | `learning_process/mlp_scratch.py`：从零实现 MLP |
| 同上（Dropout、权重衰减） | 画出「训练 vs 验证 loss」对比图 |
| `chapter_deep-learning-computation/` | 理解模型参数初始化、层与块 |

---

### 阶段 3 · 卷积神经网络与计算机视觉（⬜ 待开始）

**过关标准**：能口头解释卷积、池化的作用，能复现 LeNet / ResNet 基本结构。

| D2L 章节 | 关键内容 |
|----------|----------|
| `chapter_convolutional-neural-networks/` | 卷积、填充、池化、LeNet |
| `chapter_convolutional-modern/` | BatchNorm、ResNet、DenseNet |
| `chapter_computer-vision/`（选读） | 迁移学习、目标检测（按兴趣选章） |

**练手建议**：`learning_process/` 增加 `lenet_scratch.py`，用 Fashion-MNIST 跑通。

---

### 阶段 4 · 序列模型、注意力与 NLP（⬜ 待开始）

**过关标准**：能口头说清 RNN 梯度问题、Attention 在计算什么、为什么 Transformer 能并行。

| D2L 章节 | 关键内容 |
|----------|----------|
| `chapter_recurrent-neural-networks/` | RNN、BPTT、语言模型 |
| `chapter_recurrent-modern/` | LSTM、GRU、seq2seq、束搜索 |
| `chapter_attention-mechanisms/` | Bahdanau、多头注意力、位置编码 |
| `chapter_natural-language-processing-*`（选读） | BERT 微调等 |

> 完成本阶段后，`transformer/` 的实现会有更深理解，可回头补齐 mask、BLEU 评估。

---

### 阶段 5 · 从零实现 Transformer（🔄 持续推进）

**当前进度**：`transformer/transformer.py` 训练循环已跑通。

**下一步（按优先级）**：

1. **因果掩码**：补 `src_mask` / `tgt_mask`，防止解码器看到未来 token
2. **配置解耦**：超参数抽到 `transformer/config.py` 或命令行参数
3. **评估指标**：加 BLEU score 评估翻译质量
4. **Checkpoint**：训练中途保存/恢复模型权重
5. **笔记**：新建 `deeplearning notebook/Transformer实现.md`，记录架构设计和踩坑

**后续扩展方向**（完成上述再考虑）：

- 用 Hugging Face Transformers 微调预训练模型（对比从零实现）
- 实现 GPT-style 语言模型（纯 Decoder 架构）
- 实现 BERT 预训练 + 下游任务微调

---

## 文件命名约定

| 类型 | 格式 | 示例 |
|------|------|------|
| 个人笔记 | `deeplearning notebook/{主题}.md` | `优化算法.md` |
| 练手脚本 | `learning_process/{序号}_{主题}.py` | `03_mlp_scratch.py` |
| 独立项目 | `projects/{名称}/train.py` | `projects/sentiment/train.py` |

> 笔记里统一用「问题 + 原因 + 改法」格式记录踩坑，便于复盘。

---

## 每周节奏

| 天 | 内容 | 时间 |
|----|------|------|
| 周一、三 | 跟 D2L notebook，自己敲一遍 | 1.5~2h |
| 周二 | `learning_process/` 小脚本复现核心公式 | 1h |
| 周四 | 写/补 `deeplearning notebook/` 笔记 | 1h |
| 周五 | 改一个超参做对比实验，观察 loss 曲线变化 | 1h |
| 周六 | 推进 `transformer/` 项目或读论文 | 1~2h |
| 周日 | 休息 / 仅更新 `PROGRESS.md` | — |

---

## 环境与运行

```bash
# 验证环境
python -c "import torch; print(torch.__version__)"

# 运行 Transformer 训练
python transformer/transformer.py

# 运行线性回归练手
python learning_process/线性回归实例.py
```

**依赖**：`torch`、`d2l`、`matplotlib`
GPU 可选，CPU 完成阶段 0~2 没有问题。

---

## 遇到问题的处理顺序

1. 读完整 traceback，定位到具体文件和行号
2. `help(函数名)` 或 Jupyter `?函数名` 查文档
3. 查对应 D2L 章节的 notebook
4. 在 `deeplearning notebook/` 记一条「问题 + 原因 + 改法」
5. 仍不懂再提问，附上文件路径 + 报错内容

---

*每完成一个阶段，在 `PROGRESS.md` 顶部更新「当前阶段」一行；README 中对应状态从 ⬜ 改为 🔄 或 ✅。*
