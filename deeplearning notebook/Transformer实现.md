# Transformer 从零实现

> 对应代码：`transformer/transformer.py`
> 阶段：5（从零实现大模型）
> 状态：🔄 训练已跑通，待补 mask / checkpoint / 评估
> 理论基础对照：`注意力机制.md`

---

## 1. 整体架构

**Q：自实现的 Transformer 由哪些模块组成？数据如何在它们之间流动？**

模块清单（对照 `transformer.py`）：
- [ ] `PositionalEncoding`：正弦位置编码（`forward` 里 `x + pe`）
- [ ] `MultiHeadAttention`：W_q/W_k/W_v/W_o + 缩放点积
- [ ] `FeedForward`：两层 Linear + ReLU
- [ ] `TransformerBlock`：自注意力 + FFN，各带 `Add & Norm`
- [ ] `Encoder` / `Decoder`：堆叠 N 层
- [ ] `Transformer`：组合 encoder + decoder

数据流：
```
src → Encoder(embedding×√d + 位置编码 + N×Block) → encoder_output
tgt → Decoder(self_attn → cross_attn(用 encoder_output) → ffn) → output_linear → logits
```

> TODO：用自己的话画一张数据流图

---

## 2. 关键实现细节与踩坑

**Q（已修复）：多头注意力 `view` 后维度对不上 / `self.d_model` 缺失？**

> 记录：`MultiHeadAttention` 里 context 还原时需要 `self.d_model`（见 PROGRESS.md 阶段 5 已修复项）。
> TODO：补充当时的报错信息与定位过程

**Q：embedding 后为什么乘 $\sqrt{d_{model}}$？**

> TODO

---

## 3. 待办：因果掩码（mask）⭐

**Q：为什么解码器自注意力必须用因果掩码？`src_mask`（padding 掩码）和 `tgt_mask`（因果掩码）分别屏蔽什么？**

> 当前 `forward` 已经预留 `src_mask` / `tgt_mask` 参数，但训练时没传入。

代码骨架（自己填实现）：
```python
def make_pad_mask(seq, pad_idx=0):
    # 屏蔽 <PAD> 位置，形状广播到 [batch, 1, 1, seq_len]
    # TODO
    pass

def make_causal_mask(size):
    # 下三角矩阵，禁止看到未来 token
    # TODO（提示：torch.tril）
    pass
```

> 验收：训练时把 mask 传进 `model(src, tgt_input, src_mask, tgt_mask)`，loss 应能正常下降且推理不再「偷看未来」。

---

## 4. 待办：Checkpoint 保存与加载

```python
# 保存：torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': e}, path)
# 加载：checkpoint = torch.load(path); model.load_state_dict(checkpoint['model'])
# TODO
```

---

## 5. 待办：评估指标（BLEU）

**Q：BLEU 在衡量翻译质量的什么方面？n-gram 精度 + 长度惩罚怎么算？**

> TODO

---

## 6. 待办：配置解耦

> 把 `d_model / num_heads / num_layers / d_ff / lr` 等超参抽到 `transformer/config.py` 或命令行参数，方便做对比实验。

---

## 7. 与 D2L 实现的差异对比

> 学完 `注意力机制.md` 后回头填：自实现 vs D2L `chapter_attention-mechanisms/` 在 mask、LayerNorm 位置（Post-LN / Pre-LN）、初始化等方面的差异。

| 维度 | 我的实现 | D2L 实现 |
|------|----------|----------|
| LayerNorm 位置 | Post-LN（`norm(x + sublayer)`） | |
| Mask | （待补） | |
| 位置编码 | 正弦固定 | |

---

## 踩坑记录（问题 + 原因 + 改法）

-
