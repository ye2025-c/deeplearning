# 学习进度

> **当前阶段**：阶段 1 · 预备知识 + 线性模型（同时推进阶段 5 Transformer 项目）  
> **最后更新**：2026-05-28

在完成的任务前把 `[ ]` 改成 `[x]`。

---

## 阶段 0 · 环境与 Python

- [x] `pytorch/chapter_installation/` 环境安装完成
- [x] 能 `import torch` 并运行脚本
- [ ] Python 薄弱点清单写完（见 `深度学习初步.md` 中标注）

---

## 阶段 1 · 预备知识 + 线性模型

### 预备知识 (`chapter_preliminaries/`)

- [ ] `ndarray.ipynb`
- [ ] `pandas.ipynb`
- [x] `linear-algebra.ipynb`（遇问题再细读）
- [x] `calculus.ipynb` + `learning_process/calculus.py`
- [x] `probability.ipynb` + `learning_process/正态分布.py`
- [x] `autograd.ipynb`（笔记：`深度学习初步.md`）
- [x] `lookup-api.ipynb`（笔记：`查阅文档.md`）

### 线性模型 (`chapter_linear-networks/`)

- [x] `linear-regression.ipynb`
- [x] `linear-regression-scratch.ipynb`
- [x] `linear-regression-concise.ipynb`（笔记：`线性回归简洁实现笔记.md`，脚本：`线性回归简洁实现.py`）
- [x] `learning_process/线性回归实例.py`
- [x] 笔记：`线性神经网络.md`（开头部分）
- [ ] `softmax-regression.ipynb` 系列

### 优化 (`chapter_optimization/`)

- [ ] `optimization-intro.ipynb`
- [ ] `gd.ipynb` / `sgd.ipynb`
- [ ] 笔记：`优化算法.md`（骨架已搭，待填）

---

## 阶段 2 · 多层感知机

- [ ] `chapter_multilayer-perceptrons/` 全部 notebook
- [ ] `learning_process/` 中添加 MLP 从零实现脚本
- [ ] 笔记：`多层感知机.md`（骨架已搭，待填）
- [ ] 笔记：`深度学习计算.md`（骨架已搭，待填）

---

## 阶段 3 · 卷积与视觉

- [ ] `chapter_convolutional-neural-networks/`（笔记：`卷积神经网络.md` 骨架已搭）
- [ ] `chapter_convolutional-modern/`（笔记：`现代卷积网络.md` 骨架已搭）
- [ ] `chapter_computer-vision/`（选读，笔记：`计算机视觉.md` 骨架已搭）

---

## 阶段 4 · 序列与 NLP

- [ ] `chapter_recurrent-neural-networks/`（笔记：`循环神经网络.md` 骨架已搭）
- [ ] `chapter_recurrent-modern/`（笔记：`现代循环网络.md` 骨架已搭）
- [ ] `chapter_attention-mechanisms/`（笔记：`注意力机制.md` 骨架已搭）
- [ ] `chapter_natural-language-processing-pretraining/`（笔记：`NLP预训练.md` 骨架已搭）
- [ ] `chapter_natural-language-processing-applications/`（笔记：`NLP应用.md` 骨架已搭）

---

## 阶段 5 · 项目：Transformer

路径：`transformer/transformer.py`

- [x] 多头注意力 `self.d_model` 修复，训练可跑通
- [ ] 训练时使用 `src_mask` / `tgt_mask`（因果掩码）
- [ ] 保存与加载 checkpoint
- [ ] 笔记：`deeplearning notebook/Transformer实现.md`（骨架已搭，待填 mask/checkpoint/BLEU）
- [ ] 对比 D2L `chapter_attention-mechanisms/` 与自实现的差异

---

## 性能与工具（穿插学习）

- [x] `learning_process/Timer.py`
- [x] `learning_process/矢量加速.py`
- [ ] `chapter_computational-performance/` 相关章节（笔记：`计算性能.md` 骨架已搭）

---

## 本周计划（模板，每周一改）

| 日期 | 计划 | 完成 |
|------|------|------|
| 周一 | | [ ] |
| 周二 | | [ ] |
| 周三 | | [ ] |
| 周四 | 补笔记 | [ ] |
| 周五 | | [ ] |

---

## 笔记索引

| 文件 | 对应内容 |
|------|----------|
| `deeplearning notebook/深度学习初步.md` | DL-1，预备知识、autograd |
| `deeplearning notebook/线性神经网络.md` | DL-2，线性回归 |
| `deeplearning notebook/查阅文档.md` | lookup-api，文档查阅方法 |
| `deeplearning notebook/分离计算的作用.md` | autograd 相关，分离计算 |
| `deeplearning notebook/线性回归实例分析.md` | 线性回归原理分析 |
| `deeplearning notebook/线性回归掌握计划.md` | 线性回归学习计划 |
| `deeplearning notebook/线性回归简洁实现笔记.md` | linear-regression-concise |
| `deeplearning notebook/优化算法.md` | 阶段 1 优化章节（骨架） |
| `deeplearning notebook/多层感知机.md` | 阶段 2 MLP / 正则化（骨架） |
| `deeplearning notebook/深度学习计算.md` | 阶段 2 层与块 / 参数 / GPU（骨架） |
| `deeplearning notebook/卷积神经网络.md` | 阶段 3 CNN 基础 / LeNet（骨架） |
| `deeplearning notebook/现代卷积网络.md` | 阶段 3 AlexNet~ResNet（骨架） |
| `deeplearning notebook/计算机视觉.md` | 阶段 3 增广 / 微调 / 检测（骨架） |
| `deeplearning notebook/循环神经网络.md` | 阶段 4 序列 / RNN / BPTT（骨架） |
| `deeplearning notebook/现代循环网络.md` | 阶段 4 GRU/LSTM/seq2seq（骨架） |
| `deeplearning notebook/注意力机制.md` | 阶段 4 注意力 / Transformer（骨架） |
| `deeplearning notebook/NLP预训练.md` | 阶段 4 word2vec/GloVe/BERT（骨架） |
| `deeplearning notebook/NLP应用.md` | 阶段 4 情感分析/NLI/微调BERT（骨架） |
| `deeplearning notebook/计算性能.md` | 计算性能 多GPU/硬件（骨架） |
| `deeplearning notebook/Transformer实现.md` | 阶段 5 项目（骨架） |
