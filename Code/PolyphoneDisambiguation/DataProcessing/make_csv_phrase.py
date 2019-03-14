#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@Time：2019/3/13
@Author: hhzjj
@Description：将带多音字的注音短语按固定格式存成csv文件
"""
import pandas as pd

# 建立text 和label数组
text = []   # 存储句子
label = []  # 存储句子对应的读音


def phrase_to_csv(lines):
    for line in lines:
        line = line.strip('\n')
        line_list = line.split()
        text.append(line_list[-2])
        pron_list = line_list[-1].split(',')
        poly_list = line_list[:-2]
        for p in range(len(pron_list)):
            if str(p) not in poly_list:
                pron_list[int(p)] = 'NA'
        label.append(pron_list)
    dataframe = pd.DataFrame({'text': text, 'label': label})
    dataframe.to_csv('../data/phrase.csv', sep=',', index=False)
    print('done')


if __name__ == '__main__':
    with open('../data/poly_phrase.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [x for x in lines if x != '\n']
    phrase_to_csv(lines)
