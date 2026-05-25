# 深度学习 · 编程训练学习框架

本仓库是你的**个人深度学习训练场**：理论笔记、动手练习、教材跟练、从零实现，分层放在不同目录里，避免混在一起。

---

## 学习原则（沿用你的笔记）

1. **数学够用即可**：线性代数、微积分、概率遇到再补，不要卡在预备知识上。
2. **以框架和代码为主**：PyTorch + 《动手学深度学习》为主线；`learning_process/` 做小实验巩固。
3. **先跟练再手写**：D2L notebook 跑通 → `learning_process/` 简化复现 → `transformer/` 等大项目从零写。
4. **每学完一章就记一笔**：更新 [`PROGRESS.md`](PROGRESS.md) 里的勾选框。

---

## 目录地图

```
deeplearning/
├── README.md                 ← 本文件：总框架与路线
├── PROGRESS.md               ← 学习进度勾选表
├── learning_process/         ← 阶段 0~2：小脚本、数学与基础算法练手
├── deeplearning notebook/    ← 阶段 0~3：个人 Markdown 笔记（按 DL-章节 命名）
├── pytorch/                  ← 阶段 1~4：《动手学深度学习》PyTorch 版全套 notebook
├── transformer/              ← 阶段 5：从零实现 Transformer（进阶项目）
└── .vscode/                  ← 编辑器配置
```

| 目录 | 用途 | 何时用 |
|------|------|--------|
| `learning_process/` | 独立 `.py` 小实验（计时、微积分、线性回归等） | 想快速验证一个概念、不打开 notebook 时 |
| `deeplearning notebook/` | 自己的总结、公式推导、踩坑记录 | 每学完 D2L 一章或做完一个实验后 |
| `pytorch/` | 官方教材跟练，**不要改原 notebook 结构** | 系统跟书；用副本或新 cell 做练习 |
| `transformer/` | 完整模型实现 + 训练脚本 | 学完 RNN/注意力后再深入 |

**建议后续可增目录（需要时再建）：**

- `projects/` — 小课题（MNIST、情感分类、微调 BERT 等）
- `data/` — 自建数据集（教材数据一般由 d2l 自动下载）

---

## 学习路线（五阶段）

### 阶段 0 · 环境与 Python（约 1 周）

**目标**：能跑通 PyTorch，会看报错、会用 `help` / `dir`。

| 任务 | 材料 | 产出 |
|------|------|------|
| 安装 Python、PyTorch、d2l | `pytorch/chapter_installation/` | 环境可 `import torch` |
| Python 基础查漏补缺 | 外部教程 / 你笔记里标红的部分 | 笔记一条「已掌握列表」 |
| 第一个 tensor 实验 | `pytorch/chapter_preliminaries/ndarray.ipynb` | `PROGRESS.md` 勾选 |

**你已完成的部分**：`deeplearning notebook/深度学习初步.md`（自动微分、概率等）；`learning_process/calculus.py` 等。

---

### 阶段 1 · 预备知识 + 线性模型（约 2~3 周）

**目标**：理解张量、自动求导、线性回归 / softmax；能自己写训练循环。

| 顺序 | D2L 章节 (`pytorch/`) | 配套练手 | 笔记 |
|------|------------------------|----------|------|
| 1 | `chapter_preliminaries/`（线性代数、微积分、概率、autograd） | `learning_process/calculus.py` | `深度学习初步.md` ✓ |
| 2 | `chapter_linear-networks/` | `learning_process/线性回归实例.py` | `线性神经网络.md` ✓ |
| 3 | `chapter_optimization/`（SGD、动量等） | 自写：换学习率对比实验 | 新建 `优化算法.md` |

**阶段过关标准**：不看答案能写出「合成数据 → DataLoader → 训练 10 epoch → 打印 loss」。

---

### 阶段 2 · 多层感知机与正则（约 2 周）

| 顺序 | D2L 章节 | 练手建议 |
|------|----------|----------|
| 1 | `chapter_multilayer-perceptrons/` | 在 `learning_process/` 加 `mlp_scratch.py` |
| 2 | 同上（dropout、权重衰减） | 记录过拟合 vs 验证 loss 曲线 |

---

### 阶段 3 · 卷积与计算机视觉（约 3~4 周）

| 顺序 | D2L 章节 | 说明 |
|------|----------|------|
| 1 | `chapter_convolutional-neural-networks/` | LeNet、卷积、池化 |
| 2 | `chapter_convolutional-modern/` | ResNet、BatchNorm 等 |
| 3 | `chapter_computer-vision/` | 检测、分割等选读 |

**练手**：`learning_process/矢量加速.py` 可对照 `chapter_computational-performance/` 阅读。

---

### 阶段 4 · 序列模型与 NLP（约 3~4 周）

| 顺序 | D2L 章节 |
|------|----------|
| 1 | `chapter_recurrent-neural-networks/` |
| 2 | `chapter_recurrent-modern/`（LSTM、GRU、seq2seq） |
| 3 | `chapter_attention-mechanisms/` |
| 4 | `chapter_natural-language-processing-*`（选读） |

**阶段过关标准**：能口头说清 RNN 梯度问题、Attention 在算什么。

---

### 阶段 5 · 从零实现大模型（进行中）

| 项目 | 路径 | 状态 |
|------|------|------|
| Transformer 中英翻译 demo | `transformer/transformer.py` | 已能训练；可补 mask、BLEU、保存 checkpoint |

**下一步建议**：

1. 给训练加 `src_mask` / `tgt_mask`（因果掩码）
2. 把超参抽到 `config.py` 或命令行
3. 笔记：`deeplearning notebook/Transformer实现.md`

---

## 每周节奏（可执行模板）

| 天 | 内容 | 时间参考 |
|----|------|----------|
| 一、三 | 跟 D2L 1~2 个 notebook，**必须自己敲一遍** | 1.5~2h |
| 二 | `learning_process/` 小脚本复现核心公式 | 1h |
| 四 | 写/补 `deeplearning notebook/` 笔记 | 0.5~1h |
| 五 | 复习本周 loss 曲线、改一个超参做对比 | 1h |
| 六 | 可选：读论文节选 / 看 `transformer/` | 1~2h |
| 日 | 休息或只勾 `PROGRESS.md` | — |

---

## 文件命名约定（新建内容时）

- **笔记**：`deeplearning notebook/DL-{阶段}-{月日}-{主题}.md`（与你现有 `DL-1-3-29` 一致）
- **练手脚本**：`learning_process/{序号}_{主题}.py`，例如 `03_mlp_scratch.py`
- **项目**：`projects/{名称}/`，内含 `train.py`、`README.md`

---

## 环境

```powershell
# 建议 Python 3.10+，已安装 PyTorch 时使用：
cd e:\HuaweiMoveData\Users\shen\Desktop\deeplearning
python -c "import torch; print(torch.__version__)"

# 运行 Transformer 训练
python transformer\transformer.py

# 运行线性回归练手（需已安装 d2l）
python learning_process\线性回归实例.py
```

依赖：`torch`、`d2l`（跟 D2L 书）、`matplotlib`。GPU 可选，CPU 也能完成前期章节。

---

## 当前进度快照（2026-05）

| 模块 | 状态 |
|------|------|
| 预备知识笔记 | 进行中（`深度学习初步.md`） |
| 线性模型笔记 + 线性回归脚本 | 已开始 |
| D2L 全书 notebook | 已克隆，按 `PROGRESS.md` 逐章勾选 |
| Transformer 从零实现 | 进行中，训练已跑通 |

详细勾选见 **[`PROGRESS.md`](PROGRESS.md)**。

---

## 遇到问题时的顺序

1. 看终端完整 traceback（你已在 `transformer` 上练过）
2. `help(函数名)` 或 Jupyter `?函数名`
3. 查对应 D2L 章节 notebook
4. 在 `deeplearning notebook/` 记一条「问题 + 原因 + 改法」
5. 仍不懂再开新对话，附上文件路径和报错

---

*框架会随你进度更新；每完成一个阶段可在 `PROGRESS.md` 顶部改「当前阶段」一行。*
