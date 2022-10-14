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
    def __init__(self, name='epinions', data_pth='../data/'):
        self._g = None
        self._data_pth = data_pth

        print('=' * 20 + 'begin process' + '=' * 20)
        if self.has_cache() is False:
            self.process()
        else:
            self.load()
            print('=' * 20 + 'load graph finished' + '=' * 20)

        super(Epinions, self).__init__(name=name)

    def save(self):
        dgl.save_graphs(os.path.join(self._data_pth, 'graph.bin'), self._g)

    def read_mask(self, mode, fold=1):
        mask_pth = os.path.join(self._data_pth, f'mask4run/{mode}_fold_{fold}.npz')
        npzfile = np.load(mask_pth)
        # 构建train mask
        train_mask = torch.zeros(self._g.num_edges(mode))
        train_mask[npzfile['train']] = 1
        self._g.edges[mode].data['train_mask'] = train_mask
        # 构建test mask
        test_mask = torch.zeros(self._g.num_edges(mode))
        test_mask[npzfile['test']] = 1
        self._g.edges[mode].data['test_mask'] = test_mask

    def load(self):
        self._g = dgl.load_graphs(os.path.join(self._data_pth, 'graph.bin'))[0][0]

    def process(self):
        rate_data_name = 'rate_data.csv'
        rate_df = pd.read_csv(os.path.join(self._data_pth, rate_data_name))
        u = rate_df['user'].values
        i = rate_df['item'].values
        print('=' * 20 + 'read rate data finished' + '=' * 20)

        link_data_name = 'link_data.csv'
        link_df = pd.read_csv(os.path.join(self._data_pth, link_data_name))
        u1 = link_df['user1'].values
        u2 = link_df['user2'].values
        print('=' * 20 + 'read link data finished' + '=' * 20)

        graph_data = {
            ('user', 'rate', 'item'): (u, i),
            ('item', 'rated', 'user'): (i, u),
            ('user', 'link', 'user'): (u1, u2),
            ('user', 'linked', 'user'): (u2, u1)
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
    print(g.edges['link'].data['train_mask'])
    my_dataset.read_mask(mode='rate', fold=2)
    my_dataset.read_mask(mode='link', fold=2)
    print(g.edges['link'].data['train_mask'])