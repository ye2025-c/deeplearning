import torch
import torch.nn as nn
from d2l import torch as d2l

# 目标：复现 LeNet，在 Fashion-MNIST 上训练（建议用 GPU）
# 对应笔记：卷积神经网络.md

# 超参数
batch_size = 256
num_epochs = 10
lr = 0.9

# ── 1. 数据 ──────────────────────────────────────────────
# TODO: d2l.load_data_fashion_mnist(batch_size)


# ── 2. 模型定义 ───────────────────────────────────────────
# 结构:
#   Conv(1→6,5×5,pad=2) Sigmoid AvgPool(2)
#   Conv(6→16,5×5)      Sigmoid AvgPool(2)
#   Flatten FC(16*5*5→120) FC(120→84) FC(84→10)
def build_lenet():
    # TODO: 用 nn.Sequential 搭出上面的结构并返回
    pass


# ── 3. 设备与初始化 ───────────────────────────────────────
# TODO: device = d2l.try_gpu(); net.to(device); 权重用 xavier 初始化


# ── 4. 训练 ──────────────────────────────────────────────
# TODO: 前向 → CrossEntropyLoss → 反向 → SGD；记录 train/test acc 曲线
if __name__ == "__main__":
    pass
