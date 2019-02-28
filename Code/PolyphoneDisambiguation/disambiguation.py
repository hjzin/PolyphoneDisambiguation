# 多音字消岐
import torch
from torch import nn, optim
from torch.autograd import Variable
from DataProcessing.poly_dic import load_dict

train_data = [
    ('在古都西安', '都', 'dū'),
    ('我们都是西安人', '都', 'dōu'),
    ('西安是古都', '都', 'dū'),
    ('我们都很好', '都', 'dōu'),
]

test_data = [
    ("都是我", "都", "dōu"),
]

# 将每个字和多音字的注音编码
word_to_idx = {}
pron_to_idx = {}
for words, _, prons in train_data:
    for ch in words:
        if ch not in word_to_idx:
            word_to_idx[ch] = len(word_to_idx)
    if prons not in pron_to_idx:
        pron_to_idx[prons] = len(pron_to_idx)
print(pron_to_idx)
print(word_to_idx)

# # 读音神经网络
# class PronLSTM(nn.Module):
#     def __init__(self, n_pron, pron_dim, pron_hidden):
#         super(PronLSTM, self).__init__()
#         self.pron_embedding = nn.Embedding(n_pron, pron_dim)
#         self.pron_lstm = nn.LSTM(pron_dim, pron_hidden, batch_first=True)
#
#     def forward(self, x):
#         x = self.pron_embedding(x)
#         _, h = self.pron_lstm(x)
#         return h[0]


# 用于消岐的神经网络
class DisambiguationLSTM(nn.Module):
    def __init__(self, n_word, word_dim, word_hidden, n_pronounce):
        super(DisambiguationLSTM, self).__init__()
        self.word_embedding = nn.Embedding(n_word, word_dim)
        # self.pron_lstm = PronLSTM(n_pron, pron_dim, pron_hidden)
        self.lstm = nn.LSTM(input_size=word_dim, hidden_size=word_hidden, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(word_hidden*2, n_pronounce)

    def forward(self, x):
        x = self.word_embedding(x)
        x = x.unsqueeze(0)  # (1,5,100)
        # print('x1', x.size())
        x, _ = self.lstm(x)
        # print('x3', x.size())   # (1,5,256)
        x = self.linear1(x[:, -1, :])
        # print('x4', x.size())   # (1,2)
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
for epoch in range(100):
    print('*' * 10)
    print('eopch{}'.format(epoch + 1))
    running_loss = 0
    for data in train_data:
        word, _, pron = data
        word_list = make_sequence(word, word_to_idx)
        pron_list = [pron_to_idx[pron]]
        # print(pron_list)
        pron = Variable(torch.LongTensor(pron_list))
        # print('word_list', word_list.size())
        out = model(word_list)
        # print('out', out.size())
        loss = loss_func(out, pron)
        # print('loss', loss)
        running_loss += loss.data.numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Loss: {}'.format(running_loss / len(data)))
print()


for w, _, p in test_data:
    input_seq = make_sequence(w, word_to_idx)
    test_out = model(input_seq)
    # print(test_out)
    pred_y = torch.max(test_out, 1)[1].data.numpy()
    # print(pred_y)
    print(list(pron_to_idx.keys())[list(pron_to_idx.values()).index(pred_y[0])], 'prediction')
    print(p, 'real')


