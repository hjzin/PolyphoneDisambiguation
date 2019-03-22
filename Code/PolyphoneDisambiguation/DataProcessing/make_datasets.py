#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@Time：2019/3/20
@Author: hhzjj
@Description：划分训练集与测试集
"""
import pandas as pd
from collections import defaultdict
import DataProcessing.configure as config


class PolyCharacter:
    def __init__(self):
        self.character = ""     # 多音字
        self.num = defaultdict(int)      # 多音字每个读音出现的次数
        self.index = defaultdict(list)     # 多音字每个读音所在的句子索引


PolyPhones = []     # 存储多音字类的数组
train_data = []     # 训练集
valid_data = []     # 验证集
test_data = []      # 测试集


def find_obj_by_ch(ch):
    """
    搜索某个多音字是否已经存储过
    :param ch: 待搜索的多音字
    :return: 如果找不到，返回False和-1，如果找到，返回True和多音字对象
    """
    if len(PolyPhones) == 0:
        return False, -1
    for p in PolyPhones:
        if p.character == ch:
            return True, p
    return False, -1


def count_num_by_pron(filename):
    """
    统计每个多音字的每个读音出现的次数
    :param filename: 存储语料和读音的文件名，为csv格式
    :return: PolyPhones, text_list, new_label
    """
    df = pd.read_csv(filename)
    text_list = df['text'].tolist()
    label_list = df['label'].tolist()   # 列表存入csv时会被转成字符串，所以在读取时应将其转换成列表
    new_label = []  # 存储新的读音
    for l in label_list:
        new_label.append(list(eval(l)))

    for i in range(len(new_label)):
        for j in range(len(new_label[i])):
            if new_label[i][j] != 'NA':
                polyphone = text_list[i][j]
                pron = new_label[i][j]
                is_contain, obj = find_obj_by_ch(polyphone)
                if not is_contain:
                    polyobj = PolyCharacter()
                    polyobj.character = polyphone
                    polyobj.num[pron] += 1
                    polyobj.index[pron].append(i)
                    PolyPhones.append(polyobj)
                else:
                    obj.num[pron] += 1
                    obj.index[pron].append(i)
    return text_list, new_label


def make_datasets(trn_rate, val_rate, text, label):
    """
    划分训练集和测试集
    :param text: 所有text的集合
    :param label: 所有label的集合
    :param trn_rate: 训练集所占比例
    :param val_rate: 验证集所占比例
    :return: None
    """
    for obj in PolyPhones:
        for pron, nums in obj.num.items():
            train_len = round(trn_rate * nums)     # 训练集的长度
            val_len = round(val_rate * nums)
            for i in obj.index[pron][:train_len]:
                train_data.append(i)
            for j in obj.index[pron][train_len:train_len + val_len]:
                valid_data.append(j)
            for k in obj.index[pron][train_len + val_len:]:
                test_data.append(k)

    # 去除重复的语料
    new_train_data = list(set(train_data))
    new_train_data.sort(key=train_data.index)
    new_valid_data = list(set(valid_data))
    new_valid_data.sort(key=valid_data.index)
    new_test_data = list(set(test_data))
    new_test_data.sort(key=test_data.index)
    # 去掉三个测试集中相同的语料
    new_train_data = [x for x in new_train_data if x not in new_test_data]
    new_train_data = [x for x in new_train_data if x not in new_valid_data]
    new_valid_data = [x for x in new_valid_data if x not in new_test_data]

    print('训练集长度', len(new_train_data))
    print('测试集长度', len(new_test_data))
    print('验证集长度', len(new_valid_data))

    # 将语料写入csv文件
    write_into_csv(config.trn_file, new_train_data, text, label)
    write_into_csv(config.val_file, new_valid_data, text, label)
    write_into_csv(config.tst_file, new_test_data, text, label)


def write_into_csv(filename, data, text, label):
    """
    将分割好的语料写入csv文件
    :param filename: 文件名
    :param data: 数据列表
    :return: None
    """
    text_list = []
    label_list = []
    for d in data:
        text_list.append(text[d])
        label_list.append(label[d])
    trndata = pd.DataFrame({'text': text_list, 'label': label_list})
    trndata.to_csv(filename, sep=',', index=False)


if __name__ == '__main__':
    phrase_file = '../data/phrase.csv'
    text, label = count_num_by_pron(phrase_file)
    make_datasets(config.trn_rate, config.val_rate, text, label)

