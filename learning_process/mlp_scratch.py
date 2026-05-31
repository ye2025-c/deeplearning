import torch
from d2l import torch as d2l

# 目标：从零实现单隐藏层 MLP（Fashion-MNIST）
# 对应笔记：多层感知机.md

# 超参数
batch_size = 256
num_epochs = 10
lr = 0.1
num_inputs = 784    # 28×28
num_hiddens = 256
num_outputs = 10

# ── 1. 数据 ──────────────────────────────────────────────
# TODO: d2l.load_data_fashion_mnist(batch_size)


# ── 2. 参数初始化 ─────────────────────────────────────────
# 两层：W1 (784×256), b1 (256), W2 (256×10), b2 (10)
# TODO: 初始化参数，requires_grad=True


# ── 3. 激活函数 ───────────────────────────────────────────
def relu(X):
    # TODO: max(X, 0)
    pass


# ── 4. 模型前向 ──────────────────────────────────────────
def net(X):
    # TODO: 展平 → 隐藏层 → relu → 输出层
    pass


# ── 5. 损失函数 ───────────────────────────────────────────
# TODO: 用 nn.CrossEntropyLoss（内部含 softmax）


# ── 6. 训练循环 ──────────────────────────────────────────
# TODO: epoch 循环 → 前向 → 损失 → 反向 → SGD 更新 → 打印 train/test acc
if __name__ == "__main__":
    pass
