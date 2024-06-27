# Transformer 模型详解

Transformer 模型是一种基于**注意力机制**的深度学习模型，最初应用于自然语言处理领域，并取得了突破性的成果。如今，Transformer 模型已经广泛应用于各种领域，包括自然语言处理、计算机视觉、语音识别等。

## 1. Transformer 模型的结构

Transformer 模型主要由**编码器**和**解码器**两部分组成，如下图所示：

![f63fdc2ecb2795f26f3f7e9122deef4c](https://github.com/Einck0/NLP_Technical_Blog_JPtoCN_Transformer/assets/91471683/a8a235dc-f651-4eee-90ae-cd5d9b65aef0)


**编码器**：由多个编码器层堆叠而成，每个编码器层包含以下两个子层：

* **自注意力层（Self-Attention Layer）**：用于计算输入序列中每个词与其他词之间的关系，并生成一个新的表示向量。
* **前馈神经网络层（Feed-Forward Neural Network Layer）**：对自注意力层的输出进行非线性变换，进一步提取特征。

**解码器**：同样由多个解码器层堆叠而成，每个解码器层包含以下三个子层：

* **掩码自注意力层（Masked Self-Attention Layer）**：与编码器中的自注意力层类似，但只允许关注当前位置之前的词，防止模型在预测时“看到”未来的信息。
* **编码器-解码器注意力层（Encoder-Decoder Attention Layer）**：用于将编码器输出的信息传递给解码器，帮助解码器更好地理解输入序列。
* **前馈神经网络层（Feed-Forward Neural Network Layer）**：与编码器中的前馈神经网络层类似，对注意力层的输出进行非线性变换。

## 2. 注意力机制

注意力机制是 Transformer 模型的核心，它允许模型在处理序列数据时，**关注与当前任务最相关的部分**。

以自注意力机制为例，其计算过程如下：

1. **计算查询向量（Query）、键向量（Key）和值向量（Value）**：将输入序列中的每个词分别乘以三个不同的矩阵，得到对应的查询向量、键向量和值向量。
2. **计算注意力权重**：将查询向量与所有键向量进行点积运算，然后使用 softmax 函数进行归一化，得到每个词对应的注意力权重。
3. **加权求和**：将所有值向量乘以对应的注意力权重，然后求和，得到最终的输出向量。
![3c32a475bfb76fb84f441f7a350879d3](https://github.com/Einck0/NLP_Technical_Blog_JPtoCN_Transformer/assets/91471683/1a032954-4d12-4563-b632-7587211e045a)


## 3. Transformer 模型的优势

相比于传统的循环神经网络（RNN）模型，Transformer 模型具有以下优势：

* **并行计算**：Transformer 模型不需要像 RNN 模型那样按顺序处理序列数据，可以进行并行计算，大大提高了训练速度。
* **长距离依赖关系建模**：注意力机制允许模型直接关注距离较远的词，有效解决了 RNN 模型难以捕捉长距离依赖关系的问题。
* **可解释性强**：注意力权重可以直观地反映模型在进行预测时关注了哪些词，提高了模型的可解释性。

## 4. Transformer 模型的应用

Transformer 模型在自然语言处理领域取得了巨大的成功，例如：

* **机器翻译**：Transformer 模型在机器翻译任务上取得了显著的性能提升，例如 Google 的 BERT 模型和 OpenAI 的 GPT 模型。
* **文本生成**：Transformer 模型可以生成流畅、自然的文本，例如 OpenAI 的 GPT-3 模型。
* **问答系统**：Transformer 模型可以理解和回答用户提出的问题，例如 Google 的 BERT 模型。

除了自然语言处理领域，Transformer 模型还被应用于其他领域，例如：

* **计算机视觉**：Transformer 模型可以用于图像分类、目标检测等任务。
* **语音识别**：Transformer 模型可以用于语音识别和语音合成等任务。

## 5. 总结

Transformer 模型是一种强大的深度学习模型，其基于注意力机制的架构使其在处理序列数据方面具有显著优势。随着研究的不断深入，Transformer 模型将在更多领域发挥重要作用。  

# 基于 Transformer 和 PyTorch 的日汉机器翻译模型  
使用 Jupyter Notebook、PyTorch、Torchtext 和 SentencePiece 的教程

## 导入所需的包
首先，让我们确保在系统中安装了以下软件包，如果发现缺少某些软件包，请确保安装它们。 

```python
import math
import torchtext
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from collections import Counter
from torchtext.vocab import Vocab
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
import io
import time
import pandas as pd
import numpy as np
import pickle
import tqdm
import sentencepiece as spm
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(torch.cuda.get_device_name(0)) ## 如果你有GPU，请在你自己的电脑上尝试运行这一套代码
```


```python
torch.__version__
```




    '1.10.0+cu113'




```python
device
```




    device(type='cuda')



## 获取平行语料库
在本教程中，我们将使用从 JParaCrawl 下载的日英平行语料库！[http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl]，该语料库被描述为“由 NTT 创建的最大公开可用的英日平行语料库。它主要是通过抓取网络和自动对齐平行句子创建的”。您也可以在这里查看论文。


```python
df = pd.read_csv('./zh-ja.bicleaner05.txt', sep='\\t', engine='python', header=None)
trainen = df[2].values.tolist()#[:10000]
trainja = df[3].values.tolist()#[:10000]
# trainen.pop(5972)
# trainja.pop(5972)
```

导入所有日语及其对应的英语后，我删除了数据集中最后一个数据，因为它缺少值。trainen 和 trainja 中的句子总数为 5,973,071，但是，出于学习目的，通常建议对数据进行采样，并确保一切按预期工作，然后再一次使用所有数据，以节省时间。


以下是数据集中包含的句子示例。


```python
print(trainen[500])
print(trainja[500])
```

    Chinese HS Code Harmonized Code System < HS编码 2905 无环醇及其卤化、磺化、硝化或亚硝化衍生物 HS Code List (Harmonized System Code) for US, UK, EU, China, India, France, Japan, Russia, Germany, Korea, Canada ...
    Japanese HS Code Harmonized Code System < HSコード 2905 非環式アルコール並びにそのハロゲン化誘導体、スルホン化誘導体、ニトロ化誘導体及びニトロソ化誘導体 HS Code List (Harmonized System Code) for US, UK, EU, China, India, France, Japan, Russia, Germany, Korea, Canada ...
    

我们也可以使用不同的平行语料库来学习本文，只需确保我们可以像上面显示的那样将数据处理成两个字符串列表，分别包含日语和英语句子。

## 准备分词器
与英语或其他字母语言不同，日语句子不包含用于分隔单词的空格。我们可以使用 JParaCrawl 提供的分词器，该分词器是使用 SentencePiece 为日语和英语创建的，您可以访问 JParaCrawl 网站下载它们，或单击此处。 



```python
# 加载英语分词模型
en_tokenizer = spm.SentencePieceProcessor(model_file='spm.en.nopretok.model')

# 加载日语分词模型
ja_tokenizer = spm.SentencePieceProcessor(model_file='spm.ja.nopretok.model')
```

After the tokenizers are loaded, you can test them, for example, by executing the below code.




```python
en_tokenizer.encode("All residents aged 20 to 59 years who live in Japan must enroll in public pension system.", out_type=str)
```




    ['▁All',
     '▁residents',
     '▁aged',
     '▁20',
     '▁to',
     '▁59',
     '▁years',
     '▁who',
     '▁live',
     '▁in',
     '▁Japan',
     '▁must',
     '▁enroll',
     '▁in',
     '▁public',
     '▁pension',
     '▁system',
     '.']




```python
ja_tokenizer.encode("年金 日本に住んでいる20歳~60歳の全ての人は、公的年金制度に加入しなければなりません。", out_type=str)
```




    ['▁',
     '年',
     '金',
     '▁日本',
     'に住んでいる',
     '20',
     '歳',
     '~',
     '60',
     '歳の',
     '全ての',
     '人は',
     '、',
     '公的',
     '年',
     '金',
     '制度',
     'に',
     '加入',
     'しなければなりません',
     '。']



## 构建 TorchText 词汇表对象并将句子转换为 Torch 张量
使用分词器和原始句子，我们然后构建从 TorchText 导入的 Vocab 对象。此过程可能需要几秒钟或几分钟，具体取决于我们数据集的大小和计算能力。不同的分词器也会影响构建词汇表所需的时间，我尝试了其他几种日语分词器，但 SentencePiece 似乎运行良好且对我来说足够快。 



```python

def build_vocab(sentences, tokenizer):
    counter = Counter()  # 创建一个计数器对象
    for sentence in sentences:
        # 使用分词器对句子进行分词，并更新计数器
        counter.update(tokenizer.encode(sentence, out_type=str))
    # 创建词汇表对象，包含特定的特殊标记
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

# 使用训练集构建日语词汇表
ja_vocab = build_vocab(trainja, ja_tokenizer)
# 使用训练集构建英语词汇表
en_vocab = build_vocab(trainen, en_tokenizer)

```

在获得词汇表对象后，我们就可以使用词汇表和分词器对象为我们的训练数据构建张量。 


```python

def data_process(ja, en):
    data = []  # 初始化存储处理后数据的列表
    for (raw_ja, raw_en) in zip(ja, en):
        # 将日语句子转换为词汇索引张量
        ja_tensor_ = torch.tensor([ja_vocab[token] for token in ja_tokenizer.encode(raw_ja.rstrip("\n"), out_type=str)],
                                  dtype=torch.long)
        # 将英语句子转换为词汇索引张量
        en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer.encode(raw_en.rstrip("\n"), out_type=str)],
                                  dtype=torch.long)
        # 将处理后的张量添加到数据列表中
        data.append((ja_tensor_, en_tensor_))
    return data

# 处理训练数据
train_data = data_process(trainja, trainen)

```

## 创建 DataLoader 对象以便在训练期间进行迭代
在这里，我将 BATCH_SIZE 设置为 16 以防止出现“cuda 内存不足”，但这取决于各种因素，例如您的机器内存容量、数据大小等，因此请根据您的需要随意更改批处理大小（注意：PyTorch 的教程使用 Multi30k 德语-英语 数据集将批大小设置为 128。） 



```python
# 设置批量大小和特殊标记的索引
BATCH_SIZE = 24
PAD_IDX = ja_vocab['<pad>']
BOS_IDX = ja_vocab['<bos>']
EOS_IDX = ja_vocab['<eos>']

def generate_batch(data_batch):
    ja_batch, en_batch = [], []
    for (ja_item, en_item) in data_batch:
        # 在每个日语句子的开头和结尾添加<BOS>和<EOS>
        ja_batch.append(torch.cat([torch.tensor([BOS_IDX]), ja_item, torch.tensor([EOS_IDX])], dim=0))
        # 在每个英语句子的开头和结尾添加<BOS>和<EOS>
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    # 使用填充标记PAD_IDX对日语和英语批量数据进行填充
    ja_batch = pad_sequence(ja_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return ja_batch, en_batch

# 创建数据加载器，使用generate_batch函数生成批量数据
train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)

```

##  序列到序列 Transformer
接下来的几段代码和文字解释（用斜体表示）取自 PyTorch 官方教程 [https://pytorch.org/tutorials/beginner/translation_transformer.html](https://pytorch.org/tutorials/beginner/translation_transformer.html)。除了 BATCH_SIZE 和单词 de_vocab（已更改为 ja_vocab）之外，我没有进行任何更改。

Transformer 是“Attention is all you need”论文中介绍的一种 Seq2Seq 模型，用于解决机器翻译任务。Transformer 模型由编码器和解码器块组成，每个块包含固定数量的层。

编码器通过一系列多头注意力和前馈网络层传播输入序列来处理输入序列。编码器的输出称为内存，与目标张量一起馈送到解码器。编码器和解码器使用强制教学技术以端到端的方式进行训练。 



```python
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward:int = 512, dropout:float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        # 定义编码器层
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 定义解码器层
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 定义生成器，将编码后的表示映射到目标词汇表大小的向量
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        
        # 定义源语言和目标语言的词嵌入层
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        
        # 定义位置编码层
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        # 对源语言输入进行词嵌入和位置编码
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        
        # 对目标语言输入进行词嵌入和位置编码
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        
        # 通过编码器处理源语言输入
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        
        # 通过解码器处理目标语言输入和编码器的输出
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        
        # 将解码器的输出通过生成器映射到目标词汇表大小
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        # 对源语言输入进行词嵌入和位置编码，然后通过编码器
        return self.transformer_encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        # 对目标语言输入进行词嵌入和位置编码，然后通过解码器
        return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

```

文本标记通过使用标记嵌入来表示。位置编码被添加到标记嵌入中以引入词序的概念。



```python
import math
import torch
import torch.nn as nn
from torch import Tensor

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        # 计算位置编码的分母部分
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        # 生成位置索引
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        # 初始化位置编码矩阵
        pos_embedding = torch.zeros((maxlen, emb_size))
        # 计算位置编码的正弦部分
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        # 计算位置编码的余弦部分
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # 为位置编码矩阵增加一个维度
        pos_embedding = pos_embedding.unsqueeze(-2)

        # 定义dropout层
        self.dropout = nn.Dropout(dropout)
        # 将位置编码矩阵注册为buffer，模型保存时不会保存这个tensor的参数
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        # 将位置编码添加到token嵌入，并应用dropout
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        # 定义词嵌入层
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        # 将tokens转换为长整型并进行词嵌入，同时乘以嵌入尺寸的平方根进行缩放
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

```

我们创建了一个后续词掩码，以阻止目标词关注其后续词。我们还创建了掩码，用于屏蔽源和目标填充标记。





```python
def generate_square_subsequent_mask(sz):
    # 创建一个上三角矩阵，并将其转置，以生成下三角掩码
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    # 将掩码中的0元素填充为负无穷，1元素填充为0.0
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    # 生成目标序列的掩码
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    # 生成源序列的掩码，全为0，因为源序列不需要掩码
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    # 生成源序列和目标序列的填充掩码
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

```

Define model parameters and instantiate model. 这里我们服务器实在是计算能力有限，按照以下配置可以训练但是效果应该是不行的。如果想要看到训练的效果请使用你自己的带GPU的电脑运行这一套代码。

当你使用自己的GPU的时候，NUM_ENCODER_LAYERS 和 NUM_DECODER_LAYERS 设置为3或者更高，NHEAD设置8，EMB_SIZE设置为512。




```python

# 定义一些超参数
SRC_VOCAB_SIZE = len(ja_vocab)  # 源语言词汇表大小
TGT_VOCAB_SIZE = len(en_vocab)  # 目标语言词汇表大小
EMB_SIZE = 512  # 词嵌入维度
NHEAD = 8  # 多头注意力机制中的头数
FFN_HID_DIM = 512  # 前馈神经网络的隐藏层维度
BATCH_SIZE = 16  # 批量大小
NUM_ENCODER_LAYERS = 3  # 编码器层数
NUM_DECODER_LAYERS = 3  # 解码器层数
NUM_EPOCHS = 16  # 训练的轮数

# 初始化Seq2SeqTransformer模型
transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                 EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                 FFN_HID_DIM)

# 对模型参数进行Xavier均匀初始化
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# 将模型转移到指定设备（例如GPU）
transformer = transformer.to(device)

# 定义损失函数，忽略填充标记的索引
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# 定义优化器，使用Adam优化器
optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
)

# 定义训练一个epoch的函数
def train_epoch(model, train_iter, optimizer):
    model.train()  # 设置模型为训练模式
    losses = 0  # 初始化损失
    for idx, (src, tgt) in enumerate(train_iter):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]  # 目标输入为目标序列去掉最后一个时间步

        # 创建源和目标的掩码
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        # 模型前向传播
        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()  # 清空梯度

        tgt_out = tgt[1:, :]  # 目标输出为目标序列去掉第一个时间步
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))  # 计算损失
        loss.backward()  # 反向传播计算梯度

        optimizer.step()  # 更新模型参数
        losses += loss.item()  # 累计损失
    return losses / len(train_iter)  # 返回平均损失

# 定义评估函数
def evaluate(model, val_iter):
    model.eval()  # 设置模型为评估模式
    losses = 0  # 初始化损失
    for idx, (src, tgt) in enumerate(val_iter):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]  # 目标输入为目标序列去掉最后一个时间步

        # 创建源和目标的掩码
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        # 模型前向传播
        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]  # 目标输出为目标序列去掉第一个时间步
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))  # 计算损失
        losses += loss.item()  # 累计损失
    return losses / len(val_iter)  # 返回平均损失

```
## 开始训练
最后，在准备好必要的类和函数之后，我们就可以开始训练模型了。毫无疑问，完成训练所需的时间可能会有很大差异，这取决于很多因素，例如计算能力、参数和数据集的大小。

当我使用来自 JParaCrawl 的完整句子列表（每种语言约有 590 万个句子）训练模型时，使用单个 NVIDIA GeForce RTX 3070 GPU，每个 epoch 大约需要 5 个小时。

以下是代码：


```python
for epoch in tqdm.tqdm(range(1, NUM_EPOCHS+1)):
    start_time = time.time()
    train_loss = train_epoch(transformer, train_iter, optimizer)
    end_time = time.time()
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
          f"Epoch time = {(end_time - start_time):.3f}s"))
```

      6%|▋         | 1/16 [03:44<56:06, 224.41s/it]

    Epoch: 1, Train loss: 4.766, Epoch time = 224.414s
    

     12%|█▎        | 2/16 [07:31<52:42, 225.86s/it]

    Epoch: 2, Train loss: 3.568, Epoch time = 226.874s
    

     19%|█▉        | 3/16 [11:16<48:51, 225.52s/it]

    Epoch: 3, Train loss: 3.104, Epoch time = 225.102s
    

     25%|██▌       | 4/16 [15:03<45:13, 226.14s/it]

    Epoch: 4, Train loss: 2.781, Epoch time = 227.095s
    

     31%|███▏      | 5/16 [18:49<41:25, 225.94s/it]

    Epoch: 5, Train loss: 2.537, Epoch time = 225.582s
    

     38%|███▊      | 6/16 [22:30<37:25, 224.55s/it]

    Epoch: 6, Train loss: 2.347, Epoch time = 221.841s
    

     44%|████▍     | 7/16 [26:07<33:17, 221.90s/it]

    Epoch: 7, Train loss: 2.196, Epoch time = 216.432s
    

     50%|█████     | 8/16 [29:33<28:55, 216.92s/it]

    Epoch: 8, Train loss: 2.073, Epoch time = 206.266s
    

     56%|█████▋    | 9/16 [33:10<25:17, 216.78s/it]

    Epoch: 9, Train loss: 1.969, Epoch time = 216.475s
    

     62%|██████▎   | 10/16 [36:44<21:36, 216.04s/it]

    Epoch: 10, Train loss: 1.880, Epoch time = 214.359s
    

     69%|██████▉   | 11/16 [40:29<18:13, 218.69s/it]

    Epoch: 11, Train loss: 1.802, Epoch time = 224.712s
    

     75%|███████▌  | 12/16 [44:14<14:42, 220.67s/it]

    Epoch: 12, Train loss: 1.737, Epoch time = 225.196s
    

     81%|████████▏ | 13/16 [47:59<11:05, 221.87s/it]

    Epoch: 13, Train loss: 1.682, Epoch time = 224.622s
    

     88%|████████▊ | 14/16 [51:43<07:25, 222.65s/it]

    Epoch: 14, Train loss: 1.632, Epoch time = 224.451s
    

     94%|█████████▍| 15/16 [55:29<03:43, 223.65s/it]

    Epoch: 15, Train loss: 1.585, Epoch time = 225.945s
    

    100%|██████████| 16/16 [59:16<00:00, 222.28s/it]

    Epoch: 16, Train loss: 1.540, Epoch time = 227.039s
    

    
    
## 使用训练好的模型尝试翻译日语句子
首先，我们创建用于翻译新句子的函数，包括获取日语句子、分词、转换为张量、推理，然后将结果解码回句子等步骤，但这次是用英语。 



```python
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    # 将源序列和掩码转移到设备（如GPU）
    src = src.to(device)
    src_mask = src_mask.to(device)
    
    # 编码源序列
    memory = model.encode(src, src_mask)
    
    # 初始化解码序列，开始符号作为第一个输入
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    
    for i in range(max_len-1):
        memory = memory.to(device)
        # 创建解码序列和编码序列之间的掩码
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        # 创建目标序列的掩码
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                                    .type(torch.bool)).to(device)
        
        # 通过解码器生成输出
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        
        # 通过生成器生成下一个单词的概率分布
        prob = model.generator(out[:, -1])
        
        # 选择概率最高的单词作为下一个单词
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()
        
        # 将下一个单词添加到解码序列中
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        
        # 如果下一个单词是结束符号，则停止解码
        if next_word == EOS_IDX:
            break
    
    return ys

def translate(model, src, src_vocab, tgt_vocab, src_tokenizer):
    model.eval()  # 设置模型为评估模式
    
    # 将源语言句子编码为词汇索引序列，并添加开始和结束符号
    tokens = [BOS_IDX] + [src_vocab.stoi[tok] for tok in src_tokenizer.encode(src, out_type=str)] + [EOS_IDX]
    num_tokens = len(tokens)
    
    # 创建源序列张量和掩码
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1))
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    
    # 使用贪婪解码生成目标序列
    tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    
    # 将目标序列的词汇索引转换为对应的单词，并去除开始和结束符号
    return " ".join([tgt_vocab.itos[tok] for tok in tgt_tokens]).replace("<bos>", "").replace("<eos>", "")

```

Then, we can just call the translate function and pass the required parameters.




```python
translate(transformer, "HSコード 8515 はんだ付け用、ろう付け用又は溶接用の機器(電気式(電気加熱ガス式を含む。)", ja_vocab, en_vocab, ja_tokenizer)

```




    ' ▁H S 代 码 ▁85 15 ▁ 用 焊 接 设 备 ( 包 括 电 气 加 热 气 体 ) 。 '




```python
trainen.pop(5)
```




    'Chinese HS Code Harmonized Code System < HS编码 8515 : 电气(包括电热气体)、激光、其他光、光子束、超声波、电子束、磁脉冲或等离子弧焊接机器及装置,不论是否 HS Code List (Harmonized System Code) for US, UK, EU, China, India, France, Japan, Russia, Germany, Korea, Canada ...'




```python
trainja.pop(5)
```




    'Japanese HS Code Harmonized Code System < HSコード 8515 はんだ付け用、ろう付け用又は溶接用の機器(電気式(電気加熱ガス式を含む。)、レーザーその他の光子ビーム式、超音波式、電子ビーム式、 HS Code List (Harmonized System Code) for US, UK, EU, China, India, France, Japan, Russia, Germany, Korea, Canada ...'




```python

```

## 保存词汇表对象和训练好的模型
最后，训练完成后，我们将首先使用 Pickle 保存词汇表对象（en_vocab 和 ja_vocab）。 



```python
import pickle

# 打开一个文件，以二进制写模式存储英语词汇表
file = open('en_vocab.pkl', 'wb')
# 将英语词汇表对象写入文件
pickle.dump(en_vocab, file)
# 关闭文件
file.close()

# 打开另一个文件，以二进制写模式存储日语词汇表
file = open('ja_vocab.pkl', 'wb')
# 将日语词汇表对象写入文件
pickle.dump(ja_vocab, file)
# 关闭文件
file.close()

```

最后，我们还可以使用 PyTorch 的保存和加载函数保存模型以供以后使用。通常，根据我们以后想用模型做什么，有两种保存模型的方法。第一种方法仅用于推理，我们可以稍后加载模型并使用它将日语翻译成英语。


```python
# save model for inference
torch.save(transformer.state_dict(), 'inference_model')
```

第二种方法也用于推理，但也用于我们希望稍后加载模型并希望继续训练的情况。



```python
# save model + checkpoint to resume training later
torch.save({
    'epoch': NUM_EPOCHS,
    'model_state_dict': transformer.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': train_loss,
    }, 'model_checkpoint.tar')
```

# Conclusion
That’s it!
