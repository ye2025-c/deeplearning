# 学习进度

> **当前阶段**：阶段 1 · 预备知识 + 线性模型（同时推进阶段 5 Transformer 项目）  
> **最后更新**：2026-05-25

在完成的任务前把 `[ ]` 改成 `[x]`。

---

## 阶段 0 · 环境与 Python

- [ ] `pytorch/chapter_installation/` 环境安装完成
- [x] 能 `import torch` 并运行脚本
- [ ] Python 薄弱点清单写完（见 `深度学习初步.md` 中标注）

---

## 阶段 1 · 预备知识 + 线性模型

### 预备知识 (`chapter_preliminaries/`)

- [ ] `ndarray.ipynb`
- [ ] `pandas.ipynb`
- [ ] `linear-algebra.ipynb`（遇问题再细读）
- [ ] `calculus.ipynb` + `learning_process/calculus.py`
- [ ] `probability.ipynb` + `learning_process/正态分布.py`
- [x] `autograd.ipynb`（笔记：`深度学习初步.md`）
- [ ] `lookup-api.ipynb`

### 线性模型 (`chapter_linear-networks/`)

- [ ] `linear-regression.ipynb`
- [ ] `linear-regression-scratch.ipynb`
- [ ] `linear-regression-concise.ipynb`
- [x] `learning_process/线性回归实例.py`
- [x] 笔记：`线性神经网络.md`（开头部分）
- [ ] `softmax-regression.ipynb` 系列

### 优化 (`chapter_optimization/`)

- [ ] `optimization-intro.ipynb`
- [ ] `gd.ipynb` / `sgd.ipynb`
- [ ] 笔记：`优化算法.md`（待建）

---

## 阶段 2 · 多层感知机

- [ ] `chapter_multilayer-perceptrons/` 全部 notebook
- [ ] `learning_process/` 中添加 MLP 从零实现脚本

---

## 阶段 3 · 卷积与视觉

- [ ] `chapter_convolutional-neural-networks/`
- [ ] `chapter_convolutional-modern/`
- [ ] `chapter_computer-vision/`（选读章节自行标注）

---

## 阶段 4 · 序列与 NLP

- [ ] `chapter_recurrent-neural-networks/`
- [ ] `chapter_recurrent-modern/`
- [ ] `chapter_attention-mechanisms/`
- [ ] NLP 应用章节（选读）

---

## 阶段 5 · 项目：Transformer

路径：`transformer/transformer.py`

- [x] 多头注意力 `self.d_model` 修复，训练可跑通
- [ ] 训练时使用 `src_mask` / `tgt_mask`（因果掩码）
- [ ] 保存与加载 checkpoint
- [ ] 笔记：`deeplearning notebook/Transformer实现.md`（待建）
- [ ] 对比 D2L `chapter_attention-mechanisms/` 与自实现的差异

---

## 性能与工具（穿插学习）

- [x] `learning_process/Timer.py`
- [x] `learning_process/矢量加速.py`
- [ ] `chapter_computational-performance/` 相关章节

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
| （待建）`优化算法.md` | 阶段 1 优化章节 |
| （待建）`Transformer实现.md` | 阶段 5 项目 |
