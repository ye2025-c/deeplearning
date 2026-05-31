# 自然语言处理：预训练

> 对应章节：`pytorch/chapter_natural-language-processing-pretraining/`
> 阶段：4（序列模型、注意力与 NLP）→ 衔接阶段5 Transformer/BERT
> 状态：⬜ 待填
> 过关标准：能说清「词嵌入在学什么」「BERT 的两个预训练任务」，理解从 word2vec 到 BERT 的演进。
> 关联：`注意力机制.md`、`Transformer实现.md`

---

## 1. 词嵌入 word2vec

**Q：为什么用稠密向量表示词，而不用 one-hot？「词的意义由上下文决定」如何体现？**

要点清单：
- [ ] one-hot 的缺陷（维度高、无相似度）
- [ ] Skip-Gram：用中心词预测上下文
- [ ] CBOW：用上下文预测中心词

关键公式（Skip-Gram，softmax）：
$$
P(w_o \mid w_c) = \frac{\exp(u_o^\top v_c)}{\sum_{i \in V}\exp(u_i^\top v_c)}
$$

> TODO

---

## 2. 近似训练

**Q：词表很大时 softmax 分母计算量爆炸，负采样 / 层序 softmax 如何解决？**

要点清单：
- [ ] 负采样（Negative Sampling）：把多分类变成多个二分类
- [ ] 层序 softmax（Hierarchical Softmax）

> TODO

---

## 3. 预训练数据集与预训练 word2vec

> 练手脚本建议：`learning_process/word2vec.py`

代码骨架：
```python
# 1. 读语料 -> 建词表 -> 下采样高频词
# 2. 提取中心词 / 上下文词 / 负采样
# 3. Embedding 层 + 跳元前向（带掩码的二元交叉熵）
# 4. 训练后用余弦相似度找近义词
# TODO
```

---

## 4. GloVe（全局向量）

**Q：GloVe 如何利用「全局共现统计」？和 word2vec 的局部窗口有何不同？**

> TODO

---

## 5. 子词嵌入（fastText / BPE）

**Q：fastText 用「子词 n-gram」解决了什么（未登录词 / 形态丰富语言）？字节对编码 BPE 在做什么？**

> 注：BPE 是现代 LLM tokenizer 的基础，值得重点理解。
> TODO

---

## 6. 词相似性与类比

**Q：词向量的「king - man + woman ≈ queen」类比是怎么算出来的？**

```python
# 余弦相似度 + 向量加减
# TODO
```

---

## 7. BERT ⭐

**Q：BERT 相比 word2vec 的根本进步是什么（上下文相关 vs 静态词向量）？为什么是「双向」？**

要点清单：
- [ ] 输入表示：词元嵌入 + 段嵌入 + 位置嵌入
- [ ] 基于 Transformer Encoder（与你的 `transformer/transformer.py` 的 Encoder 对照）
- [ ] 特殊词元 `<cls>` / `<sep>`

两个预训练任务：
- [ ] **掩码语言模型（MLM）**：随机遮 15% 词元让模型还原
- [ ] **下一句预测（NSP）**：判断句子 B 是否紧接句子 A

> TODO：画出 BERT 输入构造图

---

## 8. 预训练 BERT 数据集与训练

```python
# 1. 构造 NSP 样本（一半真实下一句，一半随机句）
# 2. 构造 MLM 样本（80% <mask> / 10% 随机 / 10% 不变）
# 3. 把文本转成预训练张量
# 4. 前向 -> MLM 损失 + NSP 损失 -> 反向
# TODO
```

---

## 演进脉络速查

| 方法 | 上下文相关 | 关键思想 |
|------|-----------|----------|
| word2vec | ✗（静态） | 局部窗口预测 |
| GloVe | ✗（静态） | 全局共现统计 |
| fastText | ✗（静态） | 子词 n-gram |
| BERT | ✓（动态） | Transformer 双向 + MLM/NSP |

---

## 踩坑记录（问题 + 原因 + 改法）

-
