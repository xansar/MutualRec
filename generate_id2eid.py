#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   generate_reverse_table.py
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/11 14:22   zxx      1.0         生成节点id到边id的字典
"""

# import lib
import pandas as pd
from tqdm import tqdm
import bisect
import json

if __name__ == '__main__':
    rating_df = pd.read_csv('./data/rating_data.csv')
    user_consume_dict = {}
    for row in rating_df.itertuples():
        u = getattr(row, 'user')
        idx = getattr(row, 'Index')
        if u in user_consume_dict.keys():
            bisect.insort(user_consume_dict[u], idx)
        else:
            user_consume_dict[u] = [idx]

    print(user_consume_dict)
    with open('./data/rating_user2edge.json', 'w') as f:
        json.dump(user_consume_dict, f, indent=2)

    # link_df = pd.read_csv('./data/link_data.csv')
    # user_social_dict = {}
    # for row in link_df.itertuples():
    #     u1 = getattr(row, 'user1')
    #     u2 = getattr(row, 'user2')
    #     idx = getattr(row, 'Index')
    #     if u1 in user_social_dict.keys():
    #         bisect.insort(user_social_dict[u1], idx)
    #     else:
    #         user_social_dict[u1] = [idx]
    #
    #     if u2 in user_social_dict.keys():
    #         bisect.insort(user_social_dict[u2], idx)
    #     else:
    #         user_social_dict[u2] = [idx]
    #
    # print(user_social_dict)
    # with open('./data/link_user2edge.json', 'w') as f:
    #     json.dump(user_social_dict, f, indent=2)