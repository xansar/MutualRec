#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   metric.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/13 15:42   zxx      1.0         None
"""

# import lib
import numpy as np
import random

class BaseMetric:
    def __init__(self):
        self.init_metrics()
        self.metric_dict = dict()

    def init_metrics(self, **metric):
        self.metric_dict = dict()

    def compute_metrics(self, *input_):
        pass

    def get_batch_metrics(self, *input_):
        pass

class MutualRecMetirc(BaseMetric):
    def __init__(self, ks):
        self.ks = ks
        self.init_metrics()
        super(MutualRecMetirc, self).__init__()


    def init_metrics(self):
        self.metric_name = [
            'rate_precisions',
            'rate_recalls',
            'rate_nDCGs',
            'link_precisions',
            'link_recalls',
            'link_nDCGs',
        ]
        self.metric_dict = {}
        for m in self.metric_name:
            self.metric_dict[m] = {}
            for k in self.ks:
                self.metric_dict[m][k] = {'value': 0., 'best': 0.}

    def clear_metrics(self):
        for m in self.metric_name:
            for k in self.ks:
                self.metric_dict[m][k]['value'] = 0.

    def _precision(self, pred: list, gt: list):
        precisions = []
        recalls = []
        for k in self.ks:
            pred_k = pred[:k]
            TP = len(set(pred_k) & set(gt))
            precisions.append(TP / k)
            recalls.append(TP / len(gt))
        return precisions, recalls

    def _nDCG(self, pred: list, gt: list):
        nDCGs = []
        for k in self.ks:
            pred_k = pred[:k]
            pred_k = np.array([1 if p in gt else 0 for p in pred_k])
            factor = np.power(2, pred_k) - 1
            log_ = np.log2(np.arange(1, k + 1) + 1)
            DCG = np.sum(factor / log_)
            iDCG = np.sum(1 / log_)
            nDCG = DCG / iDCG
            nDCGs.append(nDCG)
        return nDCGs


    def compute_metrics(self, *inputs):
        rate_pred, link_pred, gt_dict = inputs
        rate_gt = gt_dict['rate']
        link_gt = gt_dict['link']
        rate_pred = rate_pred
        link_pred = link_pred

        rate_precisions = np.zeros(len(self.ks))
        rate_recalls = np.zeros(len(self.ks))
        link_precisions = np.zeros(len(self.ks))
        link_recalls = np.zeros(len(self.ks))
        rate_nDCGs = np.zeros(len(self.ks))
        link_nDCGs = np.zeros(len(self.ks))

        cnt = len(rate_pred)
        for i in range(len(rate_pred)):
            if i not in rate_gt.keys() or i not in link_gt.keys():
                cnt -= 1
                continue

            r = rate_pred[i]
            r_gt = rate_gt[i]
            cur_r_precisions, cur_r_recalls = self._precision(r, r_gt)
            cur_r_nDCGs = self._nDCG(r, r_gt)
            rate_precisions += np.array(cur_r_precisions)
            rate_recalls += np.array(cur_r_recalls)
            rate_nDCGs += np.array(cur_r_nDCGs)

            l = link_pred[i]
            l_gt = link_gt[i]
            cur_l_precisions, cur_l_recalls = self._precision(l, l_gt)
            cur_l_nDCGs = self._nDCG(l, l_gt)
            link_precisions += np.array(cur_l_precisions)
            link_recalls += np.array(cur_l_recalls)
            link_nDCGs += np.array(cur_l_nDCGs)

        for m in self.metric_name:
            for i in range(len(self.ks)):
                k = self.ks[i]
                self.metric_dict[m][k]['value'] = eval(m)[i] / cnt
                if self.metric_dict[m][k]['value'] > self.metric_dict[m][k]['best']:
                    self.metric_dict[m][k]['best'] = self.metric_dict[m][k]['value']

    def print_best_metrics(self):
        metric_str = ''
        for metric_name, k_dict in self.metric_dict.items():
            for k, v in k_dict.items():
                metric_str += f'best: {metric_name}@{k}: {v["best"]:.4f}\t'
            metric_str += '\n'
        return metric_str

class NJMMetric(BaseMetric):
    def __init__(self):
        super(NJMMetric, self).__init__()
        self.init_metrics()

    def init_metrics(self):
        self.metric_dict = {
            'rmse' : {'value': 0., 'cnt': 0, 'best': 1e8},
            'precision' : {'value': 0., 'cnt': 0, 'best': 0.},
            'recall' : {'value': 0., 'cnt': 0, 'best': 0.},
            'f1' : {'value': 0., 'cnt': 0, 'best': 0.},
        }

    def clear_metrics(self):
        for k, v in self.metric_dict.items():
            self.metric_dict[k]['value'] = 0
            self.metric_dict[k]['cnt'] = 0


    def _rmse(self, input_dict):
        prediction = input_dict['rating_prediction'][0]['rating_prediction'].cpu().detach().numpy()
        rating_pre_list = input_dict['rating_prediction'][0]['rating_pre_list'].cpu().detach().numpy()
        bsz = prediction.shape[0]
        rmse = np.sum(np.square(prediction - rating_pre_list)) / bsz
        self.metric_dict['rmse']['value'] += rmse
        self.metric_dict['rmse']['cnt'] += 1

    def _f1(self, input_dict):
        predict_link = input_dict['predict_link'].cpu().detach().numpy()
        last_pre = input_dict['last_pre']
        user_id = input_dict['user_id']
        till_record = input_dict['till_record']
        till_record_keys = input_dict['till_record_keys']
        user_node_N = input_dict['user_node_N']

        precision = 0.0
        recall = 0.0

        candidate = np.arange(user_node_N - 1)
        candidate = candidate + 1  # 从1开始，跟用户编号对应
        # viewed_link 训练集+测试集所有的好友关系列表
        viewed_link = []
        # last_pre 是从test数据集中获取的user_id的好友列表
        for user_viewed in last_pre:
            if user_viewed not in viewed_link:
                viewed_link.append(user_viewed)
        # till_record 最后一个step之前的好友关系列表
        if user_id in till_record_keys:
            for user_viewed in till_record:
                if user_viewed not in viewed_link:
                    viewed_link.append(user_viewed)
        # 所有用户中没有跟当前用户链接的用户列表
        candidate = np.array([x for x in candidate if x not in viewed_link])
        # print(len(candidate))
        # 随机抽100个负样本
        candidate = random.sample(list(candidate), 100)
        candidate_value = {}

        for user in candidate:
            # 模型对负样本的预测得分
            candidate_value[user] = predict_link[0][user]
        for user in last_pre:
            # 模型对gt的打分
            candidate_value[user] = predict_link[0][user]

        candidate_value = sorted(candidate_value.items(), key=lambda item: item[1], reverse=True)
        y_predict = []
        # 选前5个打分最高的
        for i in range(5):
            y_predict.append(candidate_value[i][0])

        tp = 0.0
        fp = 0
        if len(last_pre) < 5:
            total_ture = len(last_pre)
        else:
            total_ture = 5.0
        for y_ in y_predict:
            if y_ in last_pre:
                tp += 1.0
            else:
                fp += 1
        precision += tp / 5.0
        recall += tp / total_ture

        self.metric_dict['precision']['value'] += precision
        self.metric_dict['precision']['cnt'] += 1
        self.metric_dict['recall']['value'] += recall
        self.metric_dict['recall']['cnt'] += 1


    def compute_metric(self, input_dict, mode='rmse'):
        if mode == 'rmse':
            self._rmse(input_dict)
        elif mode == 'link':
            self._f1(input_dict)

    def get_batch_metric(self):
        for k in self.metric_dict.keys():
            if k == 'f1':
                continue
            self.metric_dict[k]['value'] /= self.metric_dict[k]['cnt']
            if k == 'rmse':
                self.metric_dict[k]['value'] = np.sqrt(self.metric_dict[k]['value'])
                if self.metric_dict[k]['value'] < self.metric_dict[k]['best']:
                    self.metric_dict[k]['best'] = self.metric_dict[k]['value']
            else:
                if self.metric_dict[k]['value'] > self.metric_dict[k]['best']:
                    self.metric_dict[k]['best'] = self.metric_dict[k]['value']
            self.metric_dict[k]['cnt'] = -1

        precision = self.metric_dict['precision']['value']
        recall = self.metric_dict['recall']['value']
        if precision != 0 and recall != 0:
            self.metric_dict['f1']['value'] = 2 * precision * recall / (precision + recall)
        if self.metric_dict['f1']['value'] > self.metric_dict['f1']['best']:
            self.metric_dict['f1']['best'] = self.metric_dict['f1']['value']
        self.metric_dict['f1']['cnt'] += -1

    def print_best_metric(self):
        metric_str = ''
        for k in self.metric_dict.keys():
            metric_str += f"best {k}: {self.metric_dict[k]['best']:.4f}\n"
        return metric_str