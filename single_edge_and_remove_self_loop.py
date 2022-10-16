#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   single_edge_and_remove_self_loop.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/14 13:01   zxx      1.0         处理社交边，两个节点之间只保留一条边，同时去除自环
"""

# import lib
import dgl
import torch
from dgl.data import DGLDataset
import pandas as pd
import os
import numpy as np

def process_link(df, name1='user1', name2='user2'):
    cpy = df.copy()
    cpy[name1], cpy[name2] = cpy[name2], cpy[name1]
    res = pd.concat([df, cpy], axis=0)
    return res.drop_duplicates().reset_index().drop(columns=['index']).copy()

if __name__ == '__main__':
    # 主要思想就是将link数据映射成为对称的邻接矩阵，然后取上三角元素
    link_df = pd.read_csv('./data/link_data.csv')
    link_df = process_link(link_df)
    user_N = 12771
    u1 = torch.tensor(link_df.user1)
    u2 = torch.tensor(link_df.user2)
    indices = torch.vstack([u1, u2])
    adj = torch.sparse_coo_tensor(indices, torch.ones(len(u1)), (user_N, user_N))
    mask = torch.triu(torch.ones(user_N, user_N), diagonal=1).to_sparse_coo()
    res = adj * mask
    to_pd = torch.vstack([res.indices(), res.values()])
    to_pd = to_pd.long().t().numpy()
    d = {
        'user1': to_pd[:, 0],
        'user2': to_pd[:, 1],
        'weight': to_pd[:, 2]
    }
    link = pd.DataFrame(d)
    print(link.describe())
    link.to_csv('./data/single_link_data.csv', index=False, header=True)