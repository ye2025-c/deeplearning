import torch
import torch.nn as nn
from torch.nn import functional as F
from d2l import torch as d2l

# 目标：实现残差块并搭一个小 ResNet，对比有/无残差连接的训练曲线
# 对应笔记：现代卷积网络.md

# 超参数
batch_size = 256
num_epochs = 10
lr = 0.05

# ── 1. 残差块 ─────────────────────────────────────────────
class Residual(nn.Module):
    def __init__(self, in_ch, out_ch, use_1x1conv=False, stride=1):
        super().__init__()
        # TODO: conv1(3×3,stride) + bn1; conv2(3×3) + bn2;
        #       可选 conv3(1×1,stride) 用于匹配通道/尺寸
        pass

    def forward(self, X):
        # TODO: Y = relu(bn1(conv1(X))); Y = bn2(conv2(Y));
        #       若有 conv3: X = conv3(X); return relu(Y + X)
        pass


# ── 2. 组装 ResNet ───────────────────────────────────────
def resnet_block(in_ch, out_ch, num_residuals, first_block=False):
    # TODO: 堆叠 num_residuals 个 Residual，第一个负责下采样（除 first_block）
    pass


def build_resnet():
    # TODO: stem(7×7 conv + bn + pool) → 多个 resnet_block → 全局池化 → FC(→10)
    pass


# ── 3. 数据 ──────────────────────────────────────────────
# TODO: d2l.load_data_fashion_mnist(batch_size, resize=96)


# ── 4. 训练 ──────────────────────────────────────────────
# TODO: 在 GPU 上训练，记录曲线；可选：去掉残差(Y+X 改成 Y)做对比
if __name__ == "__main__":
    pass
