# Japanese-Chinese Machine Translation Model with Transformer & PyTorch
A tutorial using Jupyter Notebook, PyTorch, Torchtext, and SentencePiece

## Import required packages
Firstly, let’s make sure we have the below packages installed in our system, if you found that some packages are missing, make sure to install them.


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



## Get the parallel dataset
In this tutorial, we will use the Japanese-English parallel dataset downloaded from JParaCrawl![http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl] which is described as the “largest publicly available English-Japanese parallel corpus created by NTT. It was created by largely crawling the web and automatically aligning parallel sentences.” You can also see the paper here.


```python
df = pd.read_csv('./zh-ja.bicleaner05.txt', sep='\\t', engine='python', header=None)
trainen = df[2].values.tolist()#[:10000]
trainja = df[3].values.tolist()#[:10000]
# trainen.pop(5972)
# trainja.pop(5972)
```

After importing all the Japanese and their English counterparts, I deleted the last data in the dataset because it has a missing value. In total, the number of sentences in both trainen and trainja is 5,973,071, however, for learning purposes, it is often recommended to sample the data and make sure everything is working as intended, before using all the data at once, to save time.



Here is an example of sentence contained in the dataset.




```python
print(trainen[500])
print(trainja[500])
```

    Chinese HS Code Harmonized Code System < HS编码 2905 无环醇及其卤化、磺化、硝化或亚硝化衍生物 HS Code List (Harmonized System Code) for US, UK, EU, China, India, France, Japan, Russia, Germany, Korea, Canada ...
    Japanese HS Code Harmonized Code System < HSコード 2905 非環式アルコール並びにそのハロゲン化誘導体、スルホン化誘導体、ニトロ化誘導体及びニトロソ化誘導体 HS Code List (Harmonized System Code) for US, UK, EU, China, India, France, Japan, Russia, Germany, Korea, Canada ...
    

We can also use different parallel datasets to follow along with this article, just make sure that we can process the data into the two lists of strings as shown above, containing the Japanese and English sentences.

## Prepare the tokenizers
Unlike English or other alphabetical languages, a Japanese sentence does not contain whitespaces to separate the words. We can use the tokenizers provided by JParaCrawl which was created using SentencePiece for both Japanese and English, you can visit the JParaCrawl website to download them, or click here.


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



## Build the TorchText Vocab objects and convert the sentences into Torch tensors
Using the tokenizers and raw sentences, we then build the Vocab object imported from TorchText. This process can take a few seconds or minutes depending on the size of our dataset and computing power. Different tokenizer can also affect the time needed to build the vocab, I tried several other tokenizers for Japanese but SentencePiece seems to be working well and fast enough for me.


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

After we have the vocabulary objects, we can then use the vocab and the tokenizer objects to build the tensors for our training data.




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

## Create the DataLoader object to be iterated during training
Here, I set the BATCH_SIZE to 16 to prevent “cuda out of memory”, but this depends on various things such as your machine memory capacity, size of data, etc., so feel free to change the batch size according to your needs (note: the tutorial from PyTorch sets the batch size as 128 using the Multi30k German-English dataset.)


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

## Sequence-to-sequence Transformer
The next couple of codes and text explanations (written in italic) are taken from the original PyTorch tutorial [https://pytorch.org/tutorials/beginner/translation_transformer.html]. I did not make any change except for the BATCH_SIZE and the word de_vocabwhich is changed to ja_vocab.

Transformer is a Seq2Seq model introduced in “Attention is all you need” paper for solving machine translation task. Transformer model consists of an encoder and decoder block each containing fixed number of layers.

Encoder processes the input sequence by propagating it, through a series of Multi-head Attention and Feed forward network layers. The output from the Encoder referred to as memory, is fed to the decoder along with target tensors. Encoder and decoder are trained in an end-to-end fashion using teacher forcing technique.


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

Text tokens are represented by using token embeddings. Positional encoding is added to the token embedding to introduce a notion of word order.




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

We create a subsequent word mask to stop a target word from attending to its subsequent words. We also create masks, for masking source and target padding tokens




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

## Start training
Finally, after preparing the necessary classes and functions, we are ready to train our model. This goes without saying but the time needed to finish training could vary greatly depending on a lot of things such as computing power, parameters, and size of datasets.

When I trained the model using the complete list of sentences from JParaCrawl which has around 5.9 million sentences for each language, it took around 5 hours per epoch using a single NVIDIA GeForce RTX 3070 GPU.

Here is the code:


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
    

    
    

## Try translating a Japanese sentence using the trained model
First, we create the functions to translate a new sentence, including steps such as to get the Japanese sentence, tokenize, convert to tensors, inference, and then decode the result back into a sentence, but this time in English.


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

## Save the Vocab objects and trained model
Finally, after the training has finished, we will save the Vocab objects (en_vocab and ja_vocab) first, using Pickle.


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

Lastly, we can also save the model for later use using PyTorch save and load functions. Generally, there are two ways to save the model depending what we want to use them for later. The first one is for inference only, we can load the model later and use it to translate from Japanese to English.




```python
# save model for inference
torch.save(transformer.state_dict(), 'inference_model')
```

The second one is for inference too, but also for when we want to load the model later, and want to resume the training.




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
