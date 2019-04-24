#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author: hhzjj
@Description：使用类似词性标注的方法进行多音字消岐
"""
import torch
from torch import nn, optim
from DataProcessing.preprocessing import BatchIterator
from DataProcessing import configure
from DataProcessing import num_of_polyphone
import pandas as pd


# 生成迭代器
batch_iter = BatchIterator(configure.trn_file, configure.val_file, configure.tst_file, batch_size=configure.batch_size)
train_data, valid_data, test_data = batch_iter.create_dataset()
train_iter, valid_iter, test_iter = batch_iter.get_iterator(train=train_data, valid=valid_data, test=test_data)

# 存储标注正确以及错误的语句
text_wrong = []
pron_target_wrong = []
pron_pred_wrong = []
text_correct = []
pron_target_correct = []
pron_pred_correct = []

# 开发集多音字个数
polyphone_valid = num_of_polyphone.get_num_in_phrase(configure.val_file)
# 测试集多音字个数
polyphone_tst = num_of_polyphone.get_num_in_phrase(configure.tst_file)


# 用于消岐的神经网络
class DisambiguationLSTM(nn.Module):
    def __init__(self, n_word, word_dim, word_hidden, n_pronounce):
        super(DisambiguationLSTM, self).__init__()
        self.word_embedding = nn.Embedding(n_word, word_dim)
        self.lstm = nn.LSTM(
            input_size=word_dim,
            hidden_size=word_hidden,
            num_layers=configure.num_layer,
            batch_first=True,
            bidirectional=True
        )
        self.linear1 = nn.Linear(word_hidden*2, n_pronounce)

    def forward(self, x):
        x = self.word_embedding(x)
        x, _ = self.lstm(x)     # x.size():  (1,5,256)
        x = x.squeeze(0)        # 降一维
        x = self.linear1(x)     # 此时全连接层的输入是神经网络最后一层所有time step的输出
        return x


# loss函数和优化器
model = DisambiguationLSTM(len(batch_iter.TEXT.vocab), 300, 300, len(batch_iter.LABEL.vocab))
print(model)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


def myCrossEntropy(input, target):
    """
    重新实现交叉熵函数，以便能支持batch_size大于1的情况
    :param input: 模型输出的预测值
    :param target: 标签类别
    :return: 当前的loss值
    """
    batch_size = input.size()[0]
    loss = torch.FloatTensor(1)

    for i in range(batch_size):
        loss_i = loss_func(input[i], target[i])
        loss += loss_i
    result = torch.div(loss, torch.FloatTensor([batch_size]))
    return result


def get_accuracy(texts, output, target):
    """
    求每个batch中标对的多音字数，并将标注正确和错误的句子分别输出
    :param texts: 短语文本
    :param output: 模型的输出
    :param target: 正确的读音
    :return: 该batch中标对的多音字总数
    """
    null_token = batch_iter.LABEL.vocab.stoi['NA']
    unk_token = batch_iter.LABEL.vocab.stoi['<unk>']
    pad_token = batch_iter.LABEL.vocab.stoi['<pad>']
    polyphone_index = torch.ne(target, null_token)

    batch_size = output.size()[0]
    if batch_size == 1:
        pred_y = torch.argmax(output, 1)
        pred_y = pred_y.unsqueeze(0)
    else:
        pred_y = torch.argmax(output, 2)
    num = pred_y.size()[1]  # 每个短语的字数
    correct_all = 0     # 数据集中标注正确的多音字总数
    for b in range(batch_size):
        correct_phrase = 0      # 每一个短语中标注正确的多音字数
        polyphone_num = 0       # 短语中的多音字个数
        for n in range(num):
            if polyphone_index[b][n] == 1:  # 如果该字不为非多音字
                if target[b][n] != unk_token and target[b][n] != pad_token:     # 排除<unk>和<pad>的影响
                    polyphone_num += 1
                    if pred_y[b][n] == target[b][n]:
                        correct_phrase += 1
                        correct_all += 1
        # 如果一句话中有多音字标注错误，将其输出到标注错误的文件中
        if correct_phrase != polyphone_num:
            temp_target = []
            temp_pred = []
            text_str = ""
            for n in range(num):
                text_str += batch_iter.TEXT.vocab.itos[texts[b][n]]
                temp_target.append(batch_iter.LABEL.vocab.itos[target[b][n]])
                temp_pred.append(batch_iter.LABEL.vocab.itos[pred_y[b][n]])
            text_wrong.append(text_str)
            pron_target_wrong.append(temp_target)
            pron_pred_wrong.append(temp_pred)
        else:
            # 将一句话中所有多音字均标注正确的句子输出到标注正确的文件中
            temp_target = []
            temp_pred = []
            text_str = ""
            for n in range(num):
                text_str += batch_iter.TEXT.vocab.itos[texts[b][n]]
                temp_target.append(batch_iter.LABEL.vocab.itos[target[b][n]])
                temp_pred.append(batch_iter.LABEL.vocab.itos[pred_y[b][n]])
            text_correct.append(text_str)
            pron_target_correct.append(temp_target)
            pron_pred_correct.append(temp_pred)
    return correct_all


old_acc = 0

# 训练及验证
for epoch in range(1, configure.epochs + 1):
    running_loss = 0.0
    model.train()
    print('training...')
    is_correct_train = 0
    correct_num = 0
    for step, batch in enumerate(train_iter):
        print('step: ', step)
        output = model(batch.text)
        if configure.batch_size == 1:
            target = batch.label.squeeze(0)
            loss = loss_func(output, target)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            cent_loss = myCrossEntropy(output, batch.label)
            running_loss += cent_loss.item()
            optimizer.zero_grad()
            cent_loss.backward()
            optimizer.step()
    # print(running_loss)
    train_loss = running_loss / len(train_iter)

    print('valid...')
    # model.eval()
    valid_loss = 0.0
    correct_num = 0
    for v_step, v_batch in enumerate(valid_iter):
        val_output = model(v_batch.text)
        if configure.batch_size == 1:
            val_target = v_batch.label.squeeze(0)
            val_loss = loss_func(val_output, val_target)
            valid_loss += val_loss.item()
        else:
            val_cent_loss = myCrossEntropy(val_output, v_batch.label)
            valid_loss += val_cent_loss.item()

        correct_num += get_accuracy(v_batch.text, val_output, v_batch.label)
        # 求准确率，计算公式：数据集中标对的多音字个数 / 数据集中多音字总数
    acc = correct_num / polyphone_valid
    valid_loss /= len(valid_iter)
    if acc > old_acc:
        torch.save(model.state_dict(), configure.model_path)
    old_acc = acc
    print(
        'Epoch: ', epoch,
        '|train_loss: %.4f' % train_loss,
        '|valid_loss: %.4f' % valid_loss,
        '|accuracy_val: %.4f' % acc
    )


# 在测试集上进行测试
model_test = DisambiguationLSTM(len(batch_iter.TEXT.vocab), 300, 300, len(batch_iter.LABEL.vocab))
model_test.load_state_dict(torch.load(configure.model_path))
print(model_test)
model_test.eval()
print('test...')
correct_num_tst = 0
text_wrong = []
pron_target_wrong = []
pron_pred_wrong = []
text_correct = []
pron_target_correct = []
pron_pred_correct = []
for t_step, t_batch in enumerate(test_iter):
    tst_output = model_test(t_batch.text)
    correct_num_tst += get_accuracy(t_batch.text, tst_output, t_batch.label)
acc_tst = correct_num_tst / polyphone_tst

# 将标注正确和错误的语句写入相应文件
wrong_data = pd.DataFrame({'text': text_wrong, 'target': pron_target_wrong, 'pred': pron_pred_wrong})
wrong_data.to_csv('../data/wrong_1layers_300.csv', sep=',', index=False, mode='a', encoding='utf-8')
correct_data = pd.DataFrame({'text': text_correct, 'target': pron_target_correct, 'pred':pron_pred_correct})
correct_data.to_csv('../data/correct_1layers_300.csv', sep=',', index=False, mode='a', encoding='utf-8')
print('accuracy_tst: %.4f' % acc_tst)

