# -*- coding: utf-8 -*-
# 读取多音字读音字典
import json


# 导入多音字库
with open("../data/polyphones.json", 'r', encoding='utf-8') as f3:
    load_dict = json.load(f3)
