---
name: deferred-scaffolding-todos
description: Pending scaffolding the user deferred ("后面再说") for the D2L learning repo
metadata:
  type: project
---

用户在 2026-05-31 让我搭好了 D2L 全书 ch2–15 的笔记骨架和部分练手脚本后，确认还有两类「待办」要做但**推迟到以后**（"你记住就行，后面再说"）：

1. **NLP 练手脚本骨架**：`learning_process/word2vec.py`、`sentiment_rnn.py`、`sentiment_cnn.py`（对应 NLP预训练.md / NLP应用.md）
2. **`projects/` 脚手架**：Kaggle 实战，尤其房价预测（ch4）、CIFAR-10、ImageNet Dogs；`projects/` 文件夹目前为空
3. 小缺口：D2L ch4「环境和分布偏移」概念节可顺手并进 `多层感知机.md`

**Why:** 这些是已识别但用户主动延后的缺口，不是遗忘项。
**How to apply:** 等用户提起「继续/补脚本/起项目」时再做，沿用统一骨架风格（只写注释 + 函数签名 + `# TODO`，不写实现）。不要主动催。
