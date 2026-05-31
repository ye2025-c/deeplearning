import torch
from torch.nn import functional as F
from d2l import torch as d2l

# 目标：从零实现字符级 RNN 语言模型（含梯度裁剪）
# 对应笔记：循环神经网络.md

# 超参数
batch_size = 32
num_steps = 35      # 时间步长度
num_hiddens = 512
num_epochs = 500
lr = 1

# ── 1. 数据 ──────────────────────────────────────────────
# TODO: train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)


# ── 2. 参数初始化 ─────────────────────────────────────────
def get_params(vocab_size, num_hiddens, device):
    # TODO: W_xh, W_hh, b_h, W_hq, b_q；全部 requires_grad=True
    pass


# ── 3. 初始隐状态 ─────────────────────────────────────────
def init_rnn_state(batch_size, num_hiddens, device):
    # TODO: 返回全零隐状态（注意是 tuple）
    pass


# ── 4. 单步/整段前向 ─────────────────────────────────────
def rnn(inputs, state, params):
    # TODO: 沿时间步迭代 H = tanh(X@W_xh + H@W_hh + b_h)，收集 O = H@W_hq + b_q
    pass


# ── 5. 梯度裁剪 ───────────────────────────────────────────
def grad_clipping(params, theta):
    # TODO: norm = sqrt(sum(g^2))；若 norm>theta 则按比例缩放
    pass


# ── 6. 预测（warm-up + 生成）───────────────────────────────
def predict(prefix, num_preds, net, vocab, device):
    # TODO: 先用 prefix 预热隐状态，再自回归生成 num_preds 个字符
    pass


# ── 7. 训练循环 ──────────────────────────────────────────
# TODO: 每个 epoch 顺序/随机取小批量 → 前向 → 交叉熵 → 反向 → 裁剪 → 更新；打印困惑度
if __name__ == "__main__":
    pass
