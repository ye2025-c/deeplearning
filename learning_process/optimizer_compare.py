import torch
from d2l import torch as d2l

# 目标：同数据集、同模型，只换优化器，对比 loss 曲线
# 对应笔记：优化算法.md

# 超参数
batch_size = 256
num_epochs = 10
lr = 0.05

# ── 1. 数据 ──────────────────────────────────────────────
# 可用 Fashion-MNIST 或合成线性回归数据
# TODO: 加载数据，返回 train_iter


# ── 2. 模型 ──────────────────────────────────────────────
# 简单线性层或单隐藏层 MLP（保持各优化器一致）
def build_net():
    # TODO: 返回一个新初始化的 net（每个优化器都从同样初始化开始）
    pass


# ── 3. 训练一次（给定优化器）────────────────────────────────
def train(optimizer_name):
    # TODO:
    #   1. net = build_net()
    #   2. 按 optimizer_name 构造 optimizer（'sgd' / 'momentum' / 'rmsprop' / 'adam'）
    #   3. epoch 循环：前向 → loss → 反向 → step
    #   4. 记录每个 epoch 的 loss，返回 loss 列表
    pass


# ── 4. 对比并画图 ─────────────────────────────────────────
# TODO: 对 ['sgd', 'momentum', 'rmsprop', 'adam'] 各跑一次，
#       把 loss 曲线画在同一张图上对比
if __name__ == "__main__":
    pass
