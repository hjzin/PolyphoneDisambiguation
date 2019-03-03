#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author: hhzjj
@Description：使用类似词性标注的方法进行多音字消岐
"""
import torch
from torch import nn, optim
from torch.autograd import Variable

train_data = [
    ('在古都西安', '都', ['NA', 'NA', 'dū', 'NA', 'NA']),
    ('我们都是西安人', '都', ['NA', 'NA', 'dōu', 'NA', 'NA', 'NA', 'NA']),
    ('西安是古都', '都', ['NA', 'NA', 'NA', 'NA', 'dū']),
    ('我们都很好', '都', ['NA', 'NA', 'dōu', 'NA', 'NA']),
]

test_data = [
    ('西安人都很好', '都', ['NA', 'NA', 'NA', 'dōu', 'NA', 'NA']),
]

# 将每个字和多音字的注音编码
word_to_idx = {}
pron_to_idx = {}
for words, _, prons in train_data:
    for ch in words:
        if ch not in word_to_idx:
            word_to_idx[ch] = len(word_to_idx)
    for pr in prons:
        if pr not in pron_to_idx:
            pron_to_idx[pr] = len(pron_to_idx)
print(pron_to_idx)
print(word_to_idx)


# 用于消岐的神经网络
class DisambiguationLSTM(nn.Module):
    def __init__(self, n_word, word_dim, word_hidden, n_pronounce):
        super(DisambiguationLSTM, self).__init__()
        self.word_embedding = nn.Embedding(n_word, word_dim)
        self.lstm = nn.LSTM(input_size=word_dim, hidden_size=word_hidden, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(word_hidden*2, n_pronounce)

    def forward(self, x):
        x = self.word_embedding(x)
        x = x.unsqueeze(0)      # x.size(): (1,5,100)
        x, _ = self.lstm(x)     # x.size():  (1,5,256)
        x = x.squeeze(0)        # 降一维
        x = self.linear1(x)     # 此时全连接层的输入是神经网络最后一层所有time step的输出
        return x


# loss函数和优化器
model = DisambiguationLSTM(len(word_to_idx) + 1, 100, 128, len(pron_to_idx))
print(model)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


def make_sequence(x, dic):
    idx = [dic[i] for i in x]
    idx = Variable(torch.LongTensor(idx))
    return idx


# 训练
for epoch in range(150):
    print('*' * 10)
    print('eopch{}'.format(epoch + 1))
    running_loss = 0
    for data in train_data:
        word, _, pron = data
        word_list = make_sequence(word, word_to_idx)
        pron_list = make_sequence(pron, pron_to_idx)
        out = model(word_list)
        loss = loss_func(out, pron_list)
        running_loss += loss.data.numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Loss: {}'.format(running_loss / len(data)))
print()


# 测试
for w, _, p in test_data:
    input_seq = make_sequence(w, word_to_idx)
    test_out = model(input_seq)
    print(test_out)

