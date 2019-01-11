# -*- coding: utf-8 -*-
# 数据处理
from collections import defaultdict
import json


# 计算数据中多音字的个数
def count_polyphone():
    with open("../data/pinyin.txt", 'r', encoding='utf-8') as f1:
        f1_data = f1.readlines()
    count = 0
    f1_data.pop(0)
    f1_data.pop(0)
    print("数据总长度 %d" % len(f1_data))
    with open('../data/polyphones1.txt', 'w', encoding='utf-8') as outputfile:
        for line in f1_data:
            if ',' in line:  # 多音字的每个读音会用','分割
                outputfile.write(line)
                outputfile.write('\n')
                count += 1
    print("多音字个数 %d" % count)


# 统计人民日报语料中多音字的种类及出现次数
def polyphone_frequency():
    with open("../data/198801.txt", 'r', encoding='gbk') as f:
        f_data = f.readlines()
    f_data = [x for x in f_data if x != '\n']  #去掉空行
    print('新闻条数 %d' % len(f_data))
    count = 0
    polyphones = {}  #记录多音字和出现次数
    # 将含有多音字的语料筛选出来
    with open('../data/news.txt', 'w', encoding='utf-8') as nf:
        for line in f_data:
            words = line.split()
            for w in words:
                if '{' in w:  # 多音字的注音会用'{}'标识
                    nf.write(line)
                    nf.write('\n')
                    count += 1
                    pos = w.index('{')
                    if pos == 1:
                        if w[0] in polyphones:
                            polyphones[w[0]] += 1
                        else:
                            polyphones[w[0]] = 1
                    # 可能一个词中有多个多音字
                    else:
                        for i in range(pos):
                            if w[0] in polyphones:
                                polyphones[w[0]] += 1
                            else:
                                polyphones[w[0]] = 1
    print("多音字个数 %d" % count)
    with open('../data/198801output.txt', 'w', encoding='utf-8') as f2:
        for k, v in polyphones.items():
            f2.write('%s : %d' % (k, v))
            f2.write('\n')


# 建立多音字字典，包含多音字其读音
def create_poly_dic():
    poly_dic = defaultdict(list)
    with open("../data/polyphones.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [x for x in lines if x != '\n']
    for l in lines:
        words = l.split()
        pron = words[1].split(',')
        for p in pron:
            poly_dic[words[-1]].append(p)
    # 存储成json文件
    with open("../data/polyphones.json", 'w', encoding='utf-8') as f1:
        json.dump(poly_dic, f1, ensure_ascii=False)


if __name__ == '__main__':
    count_polyphone()
    polyphone_frequency()
    create_poly_dic()