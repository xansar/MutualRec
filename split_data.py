#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   split_data.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/13 16:48   zxx      1.0         根据user2edge字典，生成train和test数据集，4折交叉验证，所以生成四组mask
"""

# import lib
import random
import numpy as np

from multiprocessing import Pool, Manager
from functools import reduce

import json
import copy

random.seed(2022)

def generate_masks(id2eid, num_folds=4):
    train_masks_lst = [[] for _ in range(num_folds)]
    test_masks_lst = [[] for _ in range(num_folds)]
    for k, v in id2eid.items():
        # 计算每一折的元素个数
        num_edges = len(v)
        len_fold = int(num_edges / num_folds)
        remaining_num = num_edges % num_folds
        length_lst = [len_fold for _ in range(num_folds)]
        for i in range(remaining_num):
            length_lst[i] += 1
        length_lst.insert(0, 0)
        for i in range(1, num_folds + 1):
            length_lst[i] += length_lst[i - 1]
        # 打乱v
        random.shuffle(v)
        # 切分
        blocks_lst = []
        for i in range(len(length_lst) - 1):
            start = length_lst[i]
            end = length_lst[i + 1]
            blocks_lst.append(v[start: end])
        # print(blocks_lst)
        assert len(blocks_lst) == num_folds
        for i in range(num_folds):
            test_masks_lst[i].extend(blocks_lst[i])
            for j in range(num_folds):
                if j != i:
                    train_masks_lst[i].extend(blocks_lst[j])
    return train_masks_lst, test_masks_lst


if __name__ == '__main__':
    num_folds = 4
    # id2eid = {
    #     i: list(range(i * 6, i * 6 + 6)) for i in range(6)
    # }
    # print(id2eid)

    # with open('./data/rate_user2edge.json', 'r') as f:
    #     id2eid = json.load(f)
    # train_masks_lst, test_masks_lst = generate_masks(id2eid, num_folds)
    # for i in range(len(train_masks_lst)):
    #     save_dict = {'train': train_masks_lst[i], 'test': test_masks_lst[i]}
    #     np.savez(f'data/mask4run/rate_fold_{i + 1}', **save_dict)
    # print(test_masks_lst)

    with open('./data/link_user2edge.json', 'r') as f:
        id2eid = json.load(f)
    train_masks_lst, test_masks_lst = generate_masks(id2eid, num_folds)
    for i in range(len(train_masks_lst)):
        save_dict = {'train': train_masks_lst[i], 'test': test_masks_lst[i]}
        np.savez(f'./data/mask4run/link_fold_{i + 1}', **save_dict)
    print(test_masks_lst)



