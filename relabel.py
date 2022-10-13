#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   relabel.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/13 16:48   zxx      1.0         None
"""

import networkx as nx
import dgl
import numpy as np
import json
import pandas as pd

def process_link(df, name1='user1', name2='user2'):
    cpy = df.copy()
    cpy[name1], cpy[name2] = cpy[name2], cpy[name1]
    res = pd.concat([df, cpy], axis=0)
    return res.drop_duplicates().reset_index().drop(columns=['index']).copy()

if __name__ == '__main__':

    rating_df = pd.read_csv('./data/raw_rating_data.csv', sep=',')
    link_df = pd.read_csv('./data/raw_link_data.csv', sep=',')

    # 保证rating和link里面的user一致
    bi_link_df = process_link(link_df)
    assert (~rating_df.user.value_counts().sort_index().index == bi_link_df.user1.value_counts().sort_index().index).sum() == 0

    # 新id
    item_N = len(rating_df.item.value_counts())
    i_array = np.arange(item_N)
    user_N = len(rating_df.user.value_counts())
    u_array = np.arange(user_N)

    # 生成字典
    sorted_user_idx_lst = sorted(rating_df.user.value_counts().index)

    user_raw_idx2new_idx = {}
    user_new_idx2raw_idx = {}
    for i in range(len(sorted_user_idx_lst)):
        raw_id = sorted_user_idx_lst[i]
        new_id = int(u_array[i])
        # print(type(raw_id), type(new_id))
        user_raw_idx2new_idx[raw_id] = new_id
        user_new_idx2raw_idx[new_id] = raw_id
        # print(user_raw_idx2new_idx)

    # 保存
    user_idx_dict = {
        'raw2new': user_raw_idx2new_idx,
        'new2raw': user_new_idx2raw_idx
    }
    with open('./data/user_idx_transfer.json', 'w') as f:
        json.dump(user_idx_dict, f, indent=2)

    # 生成字典
    sorted_item_idx_lst = sorted(rating_df.item.value_counts().index)

    item_raw_idx2new_idx = {}
    item_new_idx2raw_idx = {}
    for i in range(len(sorted_item_idx_lst)):
        raw_id = sorted_item_idx_lst[i]
        new_id = int(i_array[i])
        item_raw_idx2new_idx[raw_id] = new_id
        item_new_idx2raw_idx[new_id] = raw_id

    # 保存
    item_idx_dict = {
        'raw2new': item_raw_idx2new_idx,
        'new2raw': item_new_idx2raw_idx
    }
    with open('./data/item_idx_transfer.json', 'w') as f:
        json.dump(item_idx_dict, f, indent=2)

    # 替换原有数据
    new_rating_df = rating_df.copy()
    new_rating_df['user'] = new_rating_df['user'].apply(lambda x: user_raw_idx2new_idx[x])
    new_rating_df['item'] = new_rating_df['item'].apply(lambda x: item_raw_idx2new_idx[x])
    new_rating_df.to_csv('./data/rating_data.csv', index=False, header=True)

    new_link_df = link_df.copy()
    new_link_df['user1'] = new_link_df['user1'].apply(lambda x: user_raw_idx2new_idx[x])
    new_link_df['user2'] = new_link_df['user2'].apply(lambda x: user_raw_idx2new_idx[x])
    new_link_df.to_csv('./data/link_data.csv', index=False, header=True)