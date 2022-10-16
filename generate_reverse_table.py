#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   generate_reverse_table.py
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/11 14:22   zxx      1.0         生成倒排表
"""

# import lib
import pandas as pd
from tqdm import tqdm
import bisect
import json

if __name__ == '__main__':
    rating_df = pd.read_csv('data/rate_data.csv')
    user_consume_dict = {}
    item_consumed_dict = {}
    for row in rating_df.itertuples():
        u = getattr(row, 'user')
        i = getattr(row, 'item')
        if u in user_consume_dict.keys():
            bisect.insort(user_consume_dict[u], i)
        else:
            user_consume_dict[u] = [i]

        if i in item_consumed_dict.keys():
            bisect.insort(item_consumed_dict[i], u)
        else:
            item_consumed_dict[i] = [u]

    print(user_consume_dict)
    print(item_consumed_dict)
    with open('data/rate_reverse_table.json', 'w') as f:
        json.dump({'user_consume_dict': user_consume_dict, 'item_consumed_dict': item_consumed_dict}, f, indent=2)

    link_df = pd.read_csv('./data/single_link_data.csv')
    user_social_dict = {}
    for row in link_df.itertuples():
        u1 = getattr(row, 'user1')
        u2 = getattr(row, 'user2')
        if u1 in user_social_dict.keys():
            bisect.insort(user_social_dict[u1], u2)
        else:
            user_social_dict[u1] = [u2]

        if u2 in user_social_dict.keys():
            bisect.insort(user_social_dict[u2], u1)
        else:
            user_social_dict[u2] = [u1]

    print(user_social_dict)
    with open('./data/link_reverse_table.json', 'w') as f:
        json.dump({'user_social_dict': user_social_dict}, f, indent=2)