import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader

# ============ 位置编码 ============
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ============ 多头注意力 ============
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)
        
        output = self.W_o(context)
        return output, attn_weights

# ============ 前馈网络 ============
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# ============ Transformer块 ============
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

# ============ 编码器 ============
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(x.shape[-1])
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return x

# ============ 解码器 ============
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': MultiHeadAttention(d_model, num_heads),
                'cross_attn': MultiHeadAttention(d_model, num_heads),
                'ffn': FeedForward(d_model, d_ff),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'norm3': nn.LayerNorm(d_model),
            })
            for _ in range(num_layers)
        ])
        self.output_linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x) * math.sqrt(x.shape[-1])
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            # 自注意力
            self_attn_output, _ = layer['self_attn'](x, x, x, tgt_mask)
            x = layer['norm1'](x + self.dropout(self_attn_output))
            
            # 交叉注意力
            cross_attn_output, _ = layer['cross_attn'](x, encoder_output, encoder_output, src_mask)
            x = layer['norm2'](x + self.dropout(cross_attn_output))
            
            # FFN
            ffn_output = layer['ffn'](x)
            x = layer['norm3'](x + self.dropout(ffn_output))
        
        output = self.output_linear(x)
        return output

# ============ 完整Transformer ============
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 num_heads=8, d_ff=2048, num_layers=6, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return decoder_output

# ============ 简单数据集 ============
class SimpleDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, max_len=20):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src = self.src_texts[idx].split()[:self.max_len]
        tgt = self.tgt_texts[idx].split()[:self.max_len]
        
        src_ids = [self.src_vocab.get(w, 1) for w in src]  # 1 = <UNK>
        tgt_ids = [self.tgt_vocab.get(w, 1) for w in tgt]
        
        # 填充到最大长度
        src_ids += [0] * (self.max_len - len(src_ids))
        tgt_ids += [0] * (self.max_len - len(tgt_ids))
        
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

# ============ 主程序 ============
if __name__ == "__main__":
    # 参数设置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 256
    num_heads = 4
    num_layers = 2
    d_ff = 512
    batch_size = 2
    epochs = 5
    learning_rate = 0.001
    max_len = 10
    
    # 简单的英文->数字翻译任务（演示用）
    src_texts = [
        "hello world",
        "good morning",
        "how are you",
        "thank you very much",
        "nice to meet you"
    ]
    tgt_texts = [
        "1 2",
        "3 4",
        "5 6 7",
        "8 9 10 11",
        "12 13 14 15"
    ]
    
    # 构建词汇表
    src_words = set()
    for text in src_texts:
        src_words.update(text.split())
    src_vocab = {word: idx + 2 for idx, word in enumerate(sorted(src_words))}
    src_vocab['<PAD>'] = 0
    src_vocab['<UNK>'] = 1
    
    tgt_words = set()
    for text in tgt_texts:
        tgt_words.update(text.split())
    tgt_vocab = {word: idx + 2 for idx, word in enumerate(sorted(tgt_words))}
    tgt_vocab['<PAD>'] = 0
    tgt_vocab['<UNK>'] = 1
    
    print(f"源词汇表大小: {len(src_vocab)}")
    print(f"目标词汇表大小: {len(tgt_vocab)}")
    
    # 创建数据集和数据加载器
    dataset = SimpleDataset(src_texts, tgt_texts, src_vocab, tgt_vocab, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 模型初始化
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers
    ).to(DEVICE)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # ============ 训练 ============
    print("\n开始训练...")
    for epoch in range(epochs):
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            
            # 输入输出处理
            tgt_input = tgt[:, :-1]  # 去掉最后一个token
            tgt_output = tgt[:, 1:]  # 目标是下一个token
            
            # 前向传播
            output = model(src, tgt_input)
            
            # 计算损失（形状：[batch_size, seq_len, vocab_size]）
            loss = criterion(
                output.reshape(-1, len(tgt_vocab)),
                tgt_output.reshape(-1)
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # ============ 推理 ============
    print("\n开始推理...")
    model.eval()
    
    test_src = "hello world"
    src_ids = [src_vocab.get(w, 1) for w in test_src.split()]
    src_ids += [0] * (max_len - len(src_ids))
    src_tensor = torch.tensor([src_ids]).to(DEVICE)
    
    # 初始化目标序列（开始符号）
    tgt_ids = [tgt_vocab['<PAD>']]
    
    # 自回归生成
    with torch.no_grad():
        for _ in range(max_len - 1):
            tgt_tensor = torch.tensor([tgt_ids + [0] * (max_len - len(tgt_ids))]).to(DEVICE)
            output = model(src_tensor, tgt_tensor)
            
            # 取最后一个位置的预测结果
            next_token = output[0, len(tgt_ids) - 1].argmax(dim=-1).item()
            tgt_ids.append(next_token)
            
            # 如果生成了PAD token则停止
            if next_token == 0:
                break
    
    # 反向映射到词汇
    tgt_idx2word = {idx: word for word, idx in tgt_vocab.items()}
    predicted_text = ' '.join([tgt_idx2word.get(idx, '?') for idx in tgt_ids[1:]])
    
    print(f"\n输入: {test_src}")
    print(f"输出: {predicted_text}")
    print(f"输出token ids: {tgt_ids}")