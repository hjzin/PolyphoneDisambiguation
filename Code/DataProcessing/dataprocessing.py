# -*- coding: utf-8 -*-
# 数据处理
from collections import defaultdict
import json
from poly_dic import load_dict


# 计算多音字库中多音字的个数
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
    category = 0
    polyphones = {}  #记录多音字和出现次数
    # 将含有多音字的语料筛选出来
    with open('../data/news.txt', 'w', encoding='utf-8') as nf:
        for line in f_data:
            flag = False
            words = line.split()
            for w in words:
                if '{' in w:  # 多音字的注音会用'{}'标识
                    pos = w.index('{')
                    if pos == 1:
                        # 和已有多音字库做对比，删去字库中不存在的多音字
                        if w[0] in load_dict:
                            flag = True
                            if w[0] in polyphones:
                                polyphones[w[0]] += 1
                            else:
                                polyphones[w[0]] = 1
                    # 可能一个词中有多个多音字
                    else:
                        # 和已有多音字库做对比，删去字库中不存在的多音字
                        if w[0] in load_dict:
                            flag = True
                            for i in range(pos):
                                if w[0] in polyphones:
                                    polyphones[w[0]] += 1
                                else:
                                    polyphones[w[0]] = 1
            if flag:
                count += 1
                nf.write(line)
                nf.write('\n')
    print("包含多音字的新闻数 %d" % count)
    #将多音字按出现频次从多到少排序
    ordered_list = sorted(polyphones.items(), key=lambda item: item[1], reverse= True)
    with open('../data/198801output.txt', 'w', encoding='utf-8') as f2:
        for l in ordered_list:
                category += 1
                f2.write('%s : %d' % (l[0], l[1]))
                f2.write('\n')
    print("多音字种类数 %d" % category)


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


# 提取带注音的短语中所有含多音字的短语
def get_poly_phrase():
    with open('../data/phrase.txt', 'r', encoding='utf-8') as f:
        phrase_list = f.readlines()
    print("短语总数 %d" % len(phrase_list))
    count = 0
    ch_dic = defaultdict()
    with open("../data/poly_phrase.txt", 'w', encoding='utf-8') as of:
        for p in phrase_list:
            words = p.split('=')
            is_poly = False
            for i in range(len(words[0])):
                ch = words[0][i]
                if ch in load_dict:
                    if ch in ch_dic:
                        ch_dic[ch] += 1
                    else:
                        ch_dic[ch] = 1
                    is_poly = True
                    of.write("%d " % i)    # 在短语前输出多音字在短语中的位置
            if is_poly:
                count += 1
                of.write("%s %s" % (words[0], words[-1]))  # 输出含多音字的短语及其读音
    print("含多音字的短语总数 %d" % count)
    ordered_chlist = sorted(ch_dic.items(), key=lambda item: item[1], reverse=True)
    with open("../data/phrase_frequency.txt", 'w', encoding='utf-8') as pf:
        # 将多音字按出现频次从多到少排序
        for l in ordered_chlist:
                pf.write('%s : %d\n' % (l[0], l[1]))


if __name__ == '__main__':
    count_polyphone()
    polyphone_frequency()
    create_poly_dic()
    get_poly_phrase()