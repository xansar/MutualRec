#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   trainer.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/13 15:41   zxx      1.0         None
"""

import torch
import numpy as np
from dataset import Epinions
from tqdm import tqdm
import pickle

class Trainer:
    def __init__(self, model, loss_func, optimizer, metric, dataset: Epinions, config):
        self.config = config
        self.random_seed = eval(self.config['TRAIN']['random_seed'])
        self.log_pth = self.config['TRAIN']['log_pth'] + str(self.random_seed) + '_njm_torch.txt'
        self.print_config()
        self.dataset = dataset
        self.g = dataset[0]
        self.train_g = None
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.metric = metric
        self.data_name = config['DATA']['data_name']
        self.device = self.config['TRAIN']['device']

        self.to(self.device)

    def print_config(self):
        config_str = ''
        config_str += '=' * 10 + "Config" + '=' * 10 + '\n'
        for k, v in self.config.items():
            config_str += k + ': \n'
            for _k, _v in v.items():
                config_str += f'\t{_k}: {_v}\n'
        config_str += ('=' * 25 + '\n')
        tqdm.write(self.log(config_str, mode='w'))

    def get_pos_neg_edges(self, etype):
        u, v = self.g.edges(etype=etype)
        train_mask, test_mask = self.dataset.train_mask[etype], self.dataset.test_mask[etype]

        train_pos_u, train_pos_v = u[train_mask], v[train_mask]
        test_pos_u, test_pos_v = u[test_mask], v[test_mask]

        # neg
        adj = torch.sparse_coo_tensor(torch.vstack([u, v]), torch.ones(len(u)))
        adj_neg = 1 - adj.to_dense()
        if etype == 'link':
            adj_neg -= torch.diag_embed(torch.diag(adj_neg))
        neg_u, neg_v = torch.where(adj_neg != 0)
        neg_eids = np.random.choice(len(neg_u), g.number_of_edges(etype))
        test_size = len(test_pos_u)
        test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
        train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
        assert len(train_neg_u) == len(train_pos_u)
        return {
            'pos': {
                'train': (train_pos_u, train_pos_v),
                'test': (test_pos_u, test_pos_v),
            },
            'neg': {
                'train': (train_neg_u, train_neg_v),
                'test': (test_neg_u, test_neg_v),
            }
        }

    def generate_pos_neg_g(self, mode='train'):
        rate_dict, link_dict = self.get_pos_neg_edges('rate'), self.get_pos_neg_edges('link')
        pos_u, pos_i = rate_dict['pos'][mode]
        pos_u1, pos_u2 = link_dict['pos'][mode]
        neg_u, neg_i = rate_dict['neg'][mode]
        neg_u1, neg_u2 = link_dict['neg'][mode]
        pos_graph_data = {
            ('user', 'rate', 'item'): (pos_u, pos_i),
            ('item', 'rated-by', 'user'): (pos_i, pos_u),
            ('user', 'link', 'user'): (pos_u1, pos_u2),
        }
        neg_graph_data = {
            ('user', 'rate', 'item'): (neg_u, neg_i),
            ('item', 'rated-by', 'user'): (neg_i, neg_u),
            ('user', 'link', 'user'): (neg_u1, neg_u2),
        }
        pos_g = dgl.heterograph(pos_graph_data)
        neg_g = dgl.heterograph(neg_graph_data)
        return pos_g, neg_g

    def prepare_graph_for_train(self):
        train_g = dgl.remove_edges(self.g, self.dataset.test_mask['rate'], 'rate')
        train_g = dgl.remove_edges(train_g, self.dataset.test_mask['rate'], 'rated-by')
        train_g = dgl.remove_edges(train_g, self.dataset.test_mask['link'], 'link')
        self.train_g = train_g

    def to(self, device=None):
        if device is None:
            self.model = self.model.to(self.config['TRAIN']['device'])
            self.loss_func = self.loss_func.to(self.config['TRAIN']['device'])
        else:
            self.model = self.model.to(device)
            self.loss_func = self.loss_func.to(self.config['TRAIN']['device'])
            self.config['TRAIN']['device'] = device

    def step(self, batch_data, mode='train'):
        if mode == 'train':
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model.step(batch_data, mode=mode)
            loss = self.loss_func(output)
            loss.backward()
            self.optimizer.step()
            return loss.item()
        elif mode == 'evaluate':
            with torch.no_grad():
                self.model.eval()
                output = self.model.step(batch_data, mode=mode)
                loss = self.loss_func(output)
                self.metric.compute_metric(output)
                return loss.item()
        elif mode == 'evaluate_link':
            with torch.no_grad():
                self.model.eval()
                ipt = {'link_test_user_id': torch.tensor(batch_data['user_id'], device=self.device, dtype=torch.long)}
                # 这里输入好像只有userid，应该是直接过mlp打分
                output = self.model.step(ipt, mode='evaluate_link')
                metric_input_dict = batch_data
                metric_input_dict.update({
                    'predict_link': output['predict_link'],
                    'user_node_N': output['user_node_N']
                })
                self.metric.compute_metric(metric_input_dict, mode='link')
                return
        else:
            raise ValueError("Wrong Mode")

    def _compute_metric(self, metric_str):
        self.metric.get_batch_metric()
        for k, v in self.metric.metric_dict.items():
            metric_str += f'{k}: {self.metric.metric_dict[k]["value"]:4f}\n'
        self.metric.clear_metrics()
        return metric_str

    def log(self, str_, mode='a'):
        with open(self.log_pth, mode, encoding='utf-8') as f:
            f.write(str_)
            f.write('\n')
        return str_

    def train(self):
        tqdm.write(self.log("=" * 10 + "TRAIN BEGIN" + "=" * 10))
        epoch = eval(self.config['TRAIN']['epoch'])
        self.metric.init_metrics()
        for e in range(1, epoch + 1):
            all_loss = 0.0
            for s, batch_data in enumerate(tqdm(self.train_loader, desc='train')):
                loss = self.step(batch_data, mode='train')
                all_loss += loss

            all_loss /= s + 1
            metric_str = f'Train Epoch: {e}\nLoss: {all_loss:.4f}\n'
            if e % 1 == 0:
                all_loss = 0.0
                self.metric.clear_metrics()
                for s, batch_data in enumerate(tqdm(self.test_loader, desc='evaluate')):
                    loss = self.step(batch_data, mode='evaluate')
                    all_loss += loss

                all_loss /= s + 1

                with open("data/test_link_" + self.data_name + ".pkl", 'rb') as f:
                    test_link = pickle.load(f)
                for user_id in tqdm(test_link['last_pre'].keys(), desc='link_evaluate'):
                    # print(user_id)
                    if len(test_link['last_pre'].keys()) >= 1:
                        batch_data = {
                            'last_pre': test_link['last_pre'][user_id],
                            'user_id': user_id,
                            'till_record': test_link['till_record'][user_id],
                            'till_record_keys': test_link['till_record'].keys(),
                        }
                        self.step(batch_data, mode='evaluate_link')
                metric_str += f'Valid Epoch: {e}\n'
                metric_str += f'valid rating loss: {all_loss:.4f}\n'
                metric_str = self._compute_metric(metric_str)

                tqdm.write(self.log(metric_str))

        tqdm.write(self.log(self.metric.print_best_metric()))
        tqdm.write("=" * 10 + "TRAIN END" + "=" * 10)
