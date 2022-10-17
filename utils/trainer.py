#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   trainer.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/13 15:41   zxx      1.0         None
"""
import json

import torch
# 异常检测开启
torch.autograd.set_detect_anomaly(True)

import dgl
import numpy as np
from tqdm import tqdm
import pickle

class Trainer:
    def __init__(self, model, loss_func, optimizer, lr_reg, metric, dataset, config):
        self.config = config
        self.random_seed = eval(self.config['TRAIN']['random_seed'])
        self.log_pth = self.config['TRAIN']['log_pth'] + str(self.random_seed) + '_MutualRec.txt'
        self.print_config()
        self.dataset = dataset
        self.g = dataset[0]
        self.num_nodes_dict = {
            'user': self.g.number_of_nodes('user'),
            'item': self.g.number_of_nodes('item')
        }
        self.train_g = None
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr_reg = lr_reg
        self.metric = metric
        self.data_name = config['DATA']['data_name']
        self.device = config['TRAIN']['device']
        self.eval_step = eval(config['TRAIN']['eval_step'])

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

    def get_pos_neg_edges(self, etype, pos=True):
        u, v = self.g.edges(etype=etype)
        train_mask, test_mask = self.dataset.train_mask[etype], self.dataset.test_mask[etype]
        if pos:
            train_pos_u, train_pos_v = u[train_mask], v[train_mask]
            test_pos_u, test_pos_v = u[test_mask], v[test_mask]
            return {
                'train': (train_pos_u, train_pos_v),
                'test': (test_pos_u, test_pos_v),
            }
        else:
            # neg
            adj = torch.sparse_coo_tensor(torch.vstack([u, v]), torch.ones(len(u)))
            adj_neg = 1 - adj.to_dense()
            if etype == 'link':
                adj_neg -= torch.diag_embed(torch.diag(adj_neg))
            neg_u, neg_v = torch.where(adj_neg != 0)
            neg_eids = np.random.choice(len(neg_u), self.g.number_of_edges(etype))
            test_size = len(test_mask)
            test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
            train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
            assert len(train_neg_u) == len(train_mask)
            return {
                    'train': (train_neg_u, train_neg_v),
                    'test': (test_neg_u, test_neg_v),
                }

    def generate_pos_neg_g(self, mode='train', pos=True):
        rate_dict, link_dict = self.get_pos_neg_edges('rate', pos), self.get_pos_neg_edges('link', pos)
        if pos:
            pos_u, pos_i = rate_dict[mode]
            pos_u1, pos_u2 = link_dict[mode]
            pos_graph_data = {
                ('user', 'rate', 'item'): (pos_u, pos_i),
                ('item', 'rated', 'user'): (pos_i, pos_u),
                ('user', 'link', 'user'): (pos_u1, pos_u2),
            }
            pos_g = dgl.heterograph(pos_graph_data, num_nodes_dict=self.num_nodes_dict)
            return pos_g.to(self.device)
        else:
            neg_u, neg_i = rate_dict[mode]
            neg_u1, neg_u2 = link_dict[mode]
            neg_graph_data = {
                ('user', 'rate', 'item'): (neg_u, neg_i),
                ('item', 'rated', 'user'): (neg_i, neg_u),
                ('user', 'link', 'user'): (neg_u1, neg_u2),
            }
            neg_g = dgl.heterograph(neg_graph_data, num_nodes_dict=self.num_nodes_dict)
            return neg_g.to(self.device)

    def prepare_graph_for_train(self):
        train_g = dgl.remove_edges(self.g, self.dataset.test_mask['rate'], 'rate')
        train_g = dgl.remove_edges(train_g, self.dataset.test_mask['rate'], 'rated')
        train_g = dgl.remove_edges(train_g, self.dataset.test_mask['link'], 'link')
        return train_g.to(self.device)

    def to(self, device=None):
        if device is None:
            self.model = self.model.to(self.config['TRAIN']['device'])
            self.loss_func = self.loss_func.to(self.config['TRAIN']['device'])
        else:
            self.model = self.model.to(device)
            self.loss_func = self.loss_func.to(self.config['TRAIN']['device'])
            self.config['TRAIN']['device'] = device

    def step(self, mode='train', **inputs):
        if mode == 'train':
            self.model.train()
            self.optimizer.zero_grad()
            train_g = inputs['train_g']
            train_pos_g = inputs['train_pos_g']
            train_neg_g = inputs['train_neg_g']
            social_networks = inputs['train_social_networks']
            laplacian_lambda_max = inputs['laplacian_lambda_max']
            output = self.model(
                g=train_g,
                train_pos_g=train_pos_g,
                train_neg_g=train_neg_g,
                social_networks=social_networks,
                laplacian_lambda_max=laplacian_lambda_max
            )
            rate_loss, link_loss = self.loss_func(output)
            loss = rate_loss + link_loss
            # 反向传播时检测是否有异常值，定位code
            with torch.autograd.detect_anomaly():
                loss.backward()
            # for name, parms in self.model.named_parameters():
            #     if parms.grad is None:
            #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
            #               ' -->grad_value:', parms.grad)
            self.optimizer.step()
            self.lr_reg.step()
            return loss.item(), rate_loss.item(), link_loss.item()
        elif mode == 'evaluate':
            with torch.no_grad():
                self.model.eval()
                train_g = inputs['train_g']
                test_pos_g = inputs['test_pos_g']
                test_neg_g = inputs['test_neg_g']
                social_networks = inputs['train_social_networks']
                laplacian_lambda_max = inputs['laplacian_lambda_max']
                output = self.model.evaluate(g=train_g, test_pos_g=test_pos_g, test_neg_g=test_neg_g, social_networks=social_networks,
                                             laplacian_lambda_max=laplacian_lambda_max)
                # output里面，前4个跟训练一样 ，后两个用来计算metric
                rate_loss, link_loss = self.loss_func(output[:-2])
                loss = rate_loss + link_loss
                self.metric.compute_metrics(output[-2], output[-1], self.gt_dict)
                return loss.item(), rate_loss.item(), link_loss.item()
        else:
            raise ValueError("Wrong Mode")

    def _compute_metric(self, metric_str):
        for metric_name, k_dict in self.metric.metric_dict.items():
            for k, v in k_dict.items():
                metric_str += f'{metric_name}@{k}: {v["value"]:.4f}\t'
            metric_str += '\n'
        self.metric.clear_metrics()
        return metric_str

    def log(self, str_, mode='a'):
        with open(self.log_pth, mode, encoding='utf-8') as f:
            f.write(str_)
            f.write('\n')
        return str_

    def read_gt(self, test_pos_g):
        # 生成gt，用于计算metric
        ## rate
        rate_gt = test_pos_g.adj(etype='rate').coalesce().indices().t().tolist()
        self.gt_dict = {
            'rate': {},
            'link': {}
        }
        for pair in rate_gt:
            u, i = pair[0], pair[1]
            if u not in self.gt_dict['rate'].keys():
                self.gt_dict['rate'][u] = [i]
            else:
                self.gt_dict['rate'][u].append(i)

        ## link
        link_gt = test_pos_g.adj(etype='link').coalesce().indices().t().tolist()
        for pair in link_gt:
            u, i = pair[0], pair[1]
            if u not in self.gt_dict['link'].keys():
                self.gt_dict['link'][u] = [i]
            else:
                self.gt_dict['link'][u].append(i)


    def train(self):
        tqdm.write(self.log("=" * 10 + "TRAIN BEGIN" + "=" * 10))
        epoch = eval(self.config['TRAIN']['epoch'])
        self.metric.init_metrics()

        train_pos_g = self.generate_pos_neg_g('train', pos=True)
        test_pos_g = self.generate_pos_neg_g('test', pos=True)
        train_g = self.prepare_graph_for_train()
        train_social_networks =  dgl.edge_type_subgraph(train_g, [('user', 'link', 'user')])
        laplacian_lambda_max = torch.tensor(
            dgl.laplacian_lambda_max(train_social_networks),
            dtype=torch.float32,
            device=self.device
        )
        # 用来在测试时筛选没有在训练集中出现的物品
        self.model.generate_mask(train_g)

        self.read_gt(test_pos_g)

        for e in range(1, epoch + 1):
            train_neg_g = self.generate_pos_neg_g('train', pos=False)
            loss, rate_loss, link_loss = self.step(
                mode='train',
                train_g = train_g,
                train_pos_g = train_pos_g,
                train_neg_g = train_neg_g,
                train_social_networks = train_social_networks,
                laplacian_lambda_max = laplacian_lambda_max
            )

            metric_str = f'Train Epoch: {e}\nLoss: {loss:.4f}\trate_loss: {rate_loss:.4f}\tlink_loss: {link_loss:.4f}\n'
            if e % self.eval_step == 0:
                self.metric.clear_metrics()
                test_neg_g = self.generate_pos_neg_g('test', pos=False)
                loss, rate_loss, link_loss = self.step(
                    mode='evaluate',
                    train_g=train_g,
                    test_pos_g=test_pos_g,
                    test_neg_g=test_neg_g,
                    train_social_networks=train_social_networks,
                    laplacian_lambda_max=laplacian_lambda_max
                )
                metric_str += f'Evaluate Epoch: {e}\n'
                metric_str += f'all loss: {loss:.4f}\nrate loss: {rate_loss:.4f}\nlink loss: {link_loss:.4f}\n'
                metric_str = self._compute_metric(metric_str)

            tqdm.write(self.log(metric_str))

        tqdm.write(self.log(self.metric.print_best_metrics()))
        tqdm.write("=" * 10 + "TRAIN END" + "=" * 10)
