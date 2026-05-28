import torch
from d2l import torch as d2l

# 超参数
batch_size = 256
num_epochs = 10
lr = 0.1

# ── 1. 数据 ──────────────────────────────────────────────
# Fashion-MNIST，28×28 灰度图，10 类
# TODO: 用 d2l.load_data_fashion_mnist 加载


# ── 2. 参数初始化 ─────────────────────────────────────────
# 输入 784（28×28 展平），输出 10 类
# TODO: 初始化 W、b，记得 requires_grad=True


# ── 3. Softmax 函数 ──────────────────────────────────────
def softmax(X):
    # TODO: exp → 行求和 → 归一化
    pass


# ── 4. 模型前向 ──────────────────────────────────────────
def net(X):
    # TODO: 展平 → 线性变换 → softmax
    pass


# ── 5. 交叉熵损失 ─────────────────────────────────────────
def cross_entropy(y_hat, y):
    # TODO: 取真实类别对应的预测概率，取负对数
    pass


# ── 6. 准确率 ─────────────────────────────────────────────
def accuracy(y_hat, y):
    # TODO: argmax 取预测类别，与 y 比较
    pass


# ── 7. 优化器（手动 SGD）────────────────────────────────────
def sgd(params, lr, batch_size):
    # TODO: 参考线性回归的 sgd 实现
    pass


# ── 8. 训练循环 ──────────────────────────────────────────
# TODO: epoch 循环 → 前向 → 损失 → 反向 → 更新参数 → 打印 loss/acc
