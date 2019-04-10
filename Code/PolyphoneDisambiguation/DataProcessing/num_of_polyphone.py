#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@Time：2019/3/31
@Author: hhzjj
@Description：对训练集、测试集和验证集中的多音字个数做一些统计
"""

import DataProcessing.configure as config
from collections import defaultdict
import pandas as pd


def get_num_in_phrase(file_path):
    """
    统计各个数据集中每个短语里出现的多音字个数
    :param file_path: 数据集文件名
    :return: 1.字典:{多音字个数 : 短语个数}
             2.该数据集中多音字的总数
    """
    num_dict = defaultdict(int)
    df = pd.read_csv(file_path)
    label_list = df['label'].tolist()  # 列表存入csv时会被转成字符串，所以在读取时应将其转换成列表
    new_label = []  # 存储新的读音
    for l in label_list:
        new_label.append(list(eval(l)))
    for phrases in new_label:
        count = 0
        for pron in phrases:
            if pron != 'NA':
                count += 1
        num_dict[count] += 1    # 将该句中包含的多音字个数存入字典
    polyphone_sum = 0   # 该数据集中多音字的总数
    for k, v in num_dict.items():
        print('含有的多音字个数: %d\t 短语个数: %d' % (k, v))
        polyphone_sum += k * v
    print('多音字总数', polyphone_sum)
    return polyphone_sum


if __name__ == '__main__':
    get_num_in_phrase(config.tst_file)
