#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time：2019/3/9
@Author: hhzjj
@Description：读取多音字字典
"""
import json


# 导入多音字库
with open("../data/polyphones.json", 'r', encoding='utf-8') as f3:
    load_dict = json.load(f3)
