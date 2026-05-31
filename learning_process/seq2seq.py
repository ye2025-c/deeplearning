import torch
import torch.nn as nn
from d2l import torch as d2l

# 目标：GRU 编码器-解码器做机器翻译（带遮蔽损失 + BLEU）
# 对应笔记：现代循环网络.md

# 超参数
batch_size = 64
num_steps = 10
embed_size = 32
num_hiddens = 32
num_layers = 2
dropout = 0.1
num_epochs = 300
lr = 0.005

# ── 1. 数据 ──────────────────────────────────────────────
# TODO: train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)


# ── 2. 编码器 ─────────────────────────────────────────────
class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        # TODO: Embedding + GRU
        pass

    def forward(self, X, *args):
        # TODO: embed → GRU → 返回 (outputs, state)
        pass


# ── 3. 解码器 ─────────────────────────────────────────────
class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        # TODO: Embedding + GRU(输入拼 context) + Linear(→vocab_size)
        pass

    def init_state(self, enc_outputs, *args):
        # TODO: 用编码器最后隐状态作为初始 state
        pass

    def forward(self, X, state):
        # TODO: embed → 拼 context → GRU → Linear；返回 (logits, state)
        pass


# ── 4. 带遮蔽的损失 ───────────────────────────────────────
# TODO: 用 d2l.MaskedSoftmaxCELoss，屏蔽 <pad> 不计损失


# ── 5. 训练（teacher forcing）─────────────────────────────
# TODO: 解码器输入 = <bos> + 真实目标去掉末位；前向 → 遮蔽损失 → 反向 → 裁剪 → 更新


# ── 6. 预测 + BLEU 评估 ──────────────────────────────────
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    # TODO: 自回归生成，遇 <eos> 停止
    pass


# def bleu(pred_seq, label_seq, k): ...  # TODO: n-gram 精度 + 长度惩罚
if __name__ == "__main__":
    pass
