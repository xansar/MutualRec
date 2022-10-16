#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   dataset.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/13 16:18   zxx      1.0         None
"""

import dgl
import torch
from dgl.data import DGLDataset
import pandas as pd
import os
import numpy as np

class Epinions(DGLDataset):
    def __init__(self, name='epinions', data_pth='./data/'):
        self._g = None
        self._data_pth = data_pth
        self.train_mask = dict()
        self.test_mask = dict()

        print('=' * 20 + 'begin process' + '=' * 20)
        if self.has_cache() is False:
            self.process()
        else:
            self.load()
            print('=' * 20 + 'load graph finished' + '=' * 20)

        super(Epinions, self).__init__(name=name)

    def save(self):
        dgl.save_graphs(os.path.join(self._data_pth, 'graph.bin'), self._g)

    def read_mask(self, modes, fold=1):
        if isinstance(modes, str):
            modes = [modes]
        for mode in modes:
            mask_pth = os.path.join(self._data_pth, f'mask4run/{mode}_fold_{fold}.npz')
            npzfile = np.load(mask_pth)
            if mode == 'rate':
                # 构建train mask
                self.train_mask[mode] = torch.from_numpy(npzfile['train']).long()
                # 构建test mask
                self.test_mask[mode] = torch.from_numpy(npzfile['test']).long()
            elif mode == 'link':
                # 构建train mask
                raw = torch.from_numpy(npzfile['train'])
                extend = raw + int(self._g.num_edges(etype=mode) / 2)
                self.train_mask[mode] = torch.cat([raw, extend]).long()
                # 构建test mask
                raw = torch.from_numpy(npzfile['test'])
                extend = raw + int(self._g.num_edges(etype=mode) / 2)
                self.test_mask[mode] = torch.cat([raw, extend]).long()
            else:
                raise ValueError("Wrong Mode!!!")

    def load(self):
        self._g = dgl.load_graphs(os.path.join(self._data_pth, 'graph.bin'))[0][0]
        self.read_mask(['rate', 'link'])

    def process(self):
        rate_data_name = 'rate_data.csv'
        # rate_data_name = 'debug_rate.csv'
        rate_df = pd.read_csv(os.path.join(self._data_pth, rate_data_name))
        u = rate_df['user'].values
        i = rate_df['item'].values
        print('=' * 20 + 'read rate data finished' + '=' * 20)

        link_data_name = 'single_link_data.csv'
        # link_data_name = 'debug_link.csv'
        link_df = pd.read_csv(os.path.join(self._data_pth, link_data_name))
        u1 = link_df['user1'].values
        u2 = link_df['user2'].values
        print('=' * 20 + 'read link data finished' + '=' * 20)
        social_u1 = np.concatenate([u1, u2])
        social_u2 = np.concatenate([u2, u1])
        graph_data = {
            ('user', 'rate', 'item'): (u, i),
            ('item', 'rated-by', 'user'): (i, u),
            ('user', 'link', 'user'): (social_u1, social_u2),
        }
        self._g = dgl.heterograph(graph_data)
        self.read_mask('rate')
        self.read_mask('link')
        print('=' * 20 + 'construct graph finished' + '=' * 20)

        # 保存
        self.save()
        print('=' * 20 + 'save graph finished' + '=' * 20)

    def has_cache(self):
        if os.path.exists(os.path.join(self._data_pth, 'graph.bin')):
            return True
        else:
            return False

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        assert idx == 0
        return self._g


if __name__ == '__main__':
    my_dataset = Epinions()
    g = my_dataset[0]
    print(my_dataset.train_mask['link'][-10:])
    # print(g.edges(etype='link')[0][my_dataset.train_mask['link']][-10:])
    # print(g.edges(etype='link')[1][my_dataset.train_mask['link']][-10:])
    my_dataset.read_mask(['rate', 'link'], fold=2)
    print(my_dataset.train_mask['link'][-10:])
    # print(g.edges(etype='link')[0][my_dataset.train_mask['link']][-10:])
    # print(g.edges(etype='link')[1][my_dataset.train_mask['link']][-10:])
    my_dataset.read_mask(['rate', 'link'], fold=3)
    print(my_dataset.train_mask['link'][-10:])
    # print(g.edges(etype='link')[0][my_dataset.train_mask['link']][-10:])
    # print(g.edges(etype='link')[1][my_dataset.train_mask['link']][-10:])
    my_dataset.read_mask(['rate', 'link'], fold=4)
    print(my_dataset.train_mask['link'][-10:])
    # print(g.edges(etype='link')[0][my_dataset.train_mask['link']][-10:])
    # print(g.edges(etype='link')[1][my_dataset.train_mask['link']][-10:])