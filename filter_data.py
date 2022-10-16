#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   filter_data.py    
@Contact :   xxzhang16@fudan.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/7 13:58   zxx      1.0         None
"""

"""
按照MutualRec论文过滤数据集
标准：
    用户-过滤少于四个消费和社交记录的
    物品-少于四条评价信息的
思路：
    先过滤用户，因为用户排序会把相同的放在一起，然后在过滤物品
    待解决问题：如果过滤物品以后，导致一些用户的互动少于标准怎么办
"""
# import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_data(rating_pth, link_pth=None, sep=' '):
    """
    从文件中读取数据
    :param sep: 分隔符
    :type sep: str
    :param rating_pth: rating文件地址
    :type rating_pth: str
    :param link_pth: link文件地址
    :type link_pth: str
    :return: rating_df, link_df
    :rtype: DataFrame, DataFrame
    """
    rating_names = ['user', 'item', 'rating']
    rating_dtype = {
        'user': int,
        'item': int,
        'rating': float,
    }
    rating_df = pd.read_csv(
        rating_pth,
        skiprows=1,
        header=None,
        sep=sep,
        names=rating_names,
    )
    # 读取的时候会把最后一个空行读进来
    rating_df.drop(rating_df.tail(1).index, inplace=True)
    rating_df['user'] = rating_df['user'].astype(int)
    rating_df['item'] = rating_df['item'].astype(int)
    # 确保没有空缺值
    assert rating_df.isnull().any().sum() == 0

    if link_pth is not None:
        link_names = ['user1', 'user2', 'weight']
        link_dtype = {
            'user1': int,
            'user2': int,
            'weight': float,
        }
        link_df = pd.read_csv(
            link_pth,
            skiprows=1,
            header=None,
            sep=sep,
            names=link_names,
        )
        # 都进来发现index全部是nan，需要处理一下
        link_df = link_df.reset_index().drop(columns='index')
        # 去掉尾空行
        link_df.drop(link_df.tail(1).index, inplace=True)
        # 确保没有空缺值
        link_df['user1'] = link_df['user1'].astype(int)
        link_df['user2'] = link_df['user2'].astype(int)

        assert link_df.isnull().any().sum() == 0
        return rating_df, link_df
    else:
        return rating_df

def filter_data(rating_df_filter, link_df_filter):
    """
    对数据集进行迭代过滤
    保证：
        1. rating数据中的user和link数据中的user1，user2集合的合集完全一致
        2. rating中的user和item出现次数不少于4次，link数据中的user1，user2不少于4次
    :param link_df_filter: 社交关系表，每条信息是[user1, user2, weight]，weight=0/1
    :type link_df_filter: DataFrame
    :param rating_df_filter: 评分表，每条信息是[user, item, rating]，rating=1-5
    :type rating_df_filter: DataFrame
    :return: link_df_filter, rating_df_filter, n
    :rtype: DataFrame, DataFrame, int
    """
    # 记录迭代次数
    n = 0

    """
    while条件的意思分别是：
        (~bi_link_df_filter.user1.isin(rating_df_filter.user)).sum() link user总表中不在rating user总表中的数量
        ((~link_df_filter.user1.isin(bi_link_df_filter.user1)).sum() link.user1中不在link user总表中的数量
        ((~link_df_filter.user2.isin(bi_link_df_filter.user1)).sum() link.user2中不在link user总表中的数量
        ((~rating_df_filter.user.isin(bi_link_df_filter.user1)).sum() rating user总表不在link user总表中的数量
        ((bi_link_df_filter['user1'].value_counts() < 4).sum()  少于四个邻居的数量
        ((rating_df_filter['user'].value_counts() < 4).sum() rating.user中出现少于4次的数量
        ((rating_df_filter['item'].value_counts() < 4).sum() rating.item中出现少于4次的数量
    """
    # 每条边正反都录入一次，把有向变无向
    bi_link_df_filter = process_link(link_df_filter)

    while ((~bi_link_df_filter.user1.isin(rating_df_filter.user)).sum() >= 1) or \
            (link_df_filter.user1 == link_df_filter.user2).sum() >= 1 or\
        ((~link_df_filter.user1.isin(bi_link_df_filter.user1)).sum() >= 1) or \
        ((~link_df_filter.user2.isin(bi_link_df_filter.user1)).sum() >= 1) or \
        ((~rating_df_filter.user.isin(bi_link_df_filter.user1)).sum() >= 1) or \
        ((bi_link_df_filter['user1'].value_counts() < 4).sum() >= 1) or \
        ((rating_df_filter['user'].value_counts() < 4).sum() >= 1) or \
        ((rating_df_filter['item'].value_counts() < 4).sum() >= 1):

        # # 去掉自环
        # bi_link_df_filter = bi_link_df_filter[~(bi_link_df_filter.user1 == bi_link_df_filter.user2)]
        # 所有用户至少有四个邻居
        bi_link_df_filter = bi_link_df_filter[bi_link_df_filter.groupby('user1').user1.transform('count') >= 4]


        # 将rating中user，item出现少于四次的过滤掉
        rating_df_filter = rating_df_filter[rating_df_filter.groupby('user').user.transform('count') >= 4]
        rating_df_filter = rating_df_filter[rating_df_filter.groupby('item').item.transform('count') >= 4]

        # 将在rating user表，不在link user表的过滤掉
        rating_df_filter = rating_df_filter[rating_df_filter.user.isin(bi_link_df_filter.user1)]

        # 将在link user表，不在rating user表的过滤掉
        bi_link_df_filter = bi_link_df_filter[bi_link_df_filter.user1.isin(rating_df_filter.user)]

        # 将不在link总表的过滤掉
        link_df_filter = link_df_filter[link_df_filter.user1.isin(bi_link_df_filter.user1)]
        link_df_filter = link_df_filter[link_df_filter.user2.isin(bi_link_df_filter.user1)]
        link_df_filter = link_df_filter[~(link_df_filter.user1 == link_df_filter.user2)]

        # 更新无向边link表
        bi_link_df_filter = process_link(link_df_filter)

        n += 1
        print(
            ((~bi_link_df_filter.user1.isin(rating_df_filter.user)).sum()),
            (link_df_filter.user1 == link_df_filter.user2).sum(),
            ((~link_df_filter.user1.isin(bi_link_df_filter.user1)).sum()),
            ((~link_df_filter.user2.isin(bi_link_df_filter.user1)).sum()),
            ((~rating_df_filter.user.isin(bi_link_df_filter.user1)).sum()),
            ((bi_link_df_filter['user1'].value_counts() < 4).sum()),
            ((rating_df_filter['user'].value_counts() < 4).sum()),
            ((rating_df_filter['item'].value_counts() < 4).sum())
        )
        if n > 100:
            break
    return rating_df_filter, link_df_filter, n

def draw_freq_dist_pic(df, name1, name2, step=100):
    name1_freq_series = df[name1].value_counts().sort_values(ascending=False)
    name1_freq = list(name1_freq_series.values)[0::step]

    name2_freq_series = df[name2].value_counts().sort_values(ascending=False)
    name2_freq = list(name2_freq_series.values)[0::step]

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title(name1 + ' freq')
    x_lim_num = [0, 25, 50, 75, 100, 125]
    x_ticks = ['0', '2500', '5000', '7500', '10000', '12500']
    ax.bar(list(range(1, len(name1_freq) + 1)), name1_freq)
    ax.set_xticks(x_lim_num, x_ticks)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title(name2 + ' freq')
    # ax.set_ylim((0, 1500))
    x_lim_num = [0, 50, 100, 150, 200]
    x_ticks = ['0', '5000', '10000', '15000', '20000']
    ax.bar(range(1, len(name2_freq) + 1), name2_freq)
    ax.set_xticks(x_lim_num, x_ticks)

    plt.show()

def process_link(df, name1='user1', name2='user2'):
    cpy = df.copy()
    cpy[name1], cpy[name2] = cpy[name2], cpy[name1]
    res = pd.concat([df, cpy], axis=0)
    return res.drop_duplicates().reset_index().drop(columns=['index']).copy()

if __name__ == '__main__':
    rating_df, link_df = read_data('../dataset/ratings_data.txt', '../dataset/trust_data.txt')
    print(link_df.describe())

    rating_df, link_df, n = filter_data(rating_df, link_df)
    print(n)
    rating_df.to_csv('./data/raw_rate_data.csv', index=False, header=True)
    link_df.to_csv('./data/raw_link_data.csv', index=False, header=True)


# rating_df = pd.read_csv('./data/rate_data.csv')
# draw_freq_dist_pic(rating_df, 'user', 'item')

    link_df = pd.read_csv('./data/link_data.csv')
    tmp = pd.concat([link_df.user1, link_df.user2], axis=0).value_counts().sort_values(ascending=False)
    print(tmp.head())
    freqs = list(tmp.values)[0::50]
    x = list(range(1, len(freqs) + 1))
    x_lim_num = [0, 50, 100, 150, 200, 250]
    x_ticks = ['0', '2500', '5000', '7500', '10000', '12500']
    plt.bar(x, freqs)
    plt.title('user social freq')
    plt.xticks(x_lim_num, x_ticks)
    plt.show()