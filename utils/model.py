#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   model.py
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/13 12:51   zxx      1.0         None
"""

import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
import torch
import torch.nn as nn
import numpy as np
import random

class SpatialAttentionLayer_GAT(nn.Module):
    def __init__(self, embedding_size=1, num_heads=1, rel_names=None):
        super(SpatialAttentionLayer_GAT, self).__init__()
        if rel_names is None:
            rel_names = ['rate', 'rated-by', 'link']
        self.gat_layer_1 = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GATv2Conv(embedding_size, embedding_size, num_heads=num_heads)
                for rel in rel_names
            },
            aggregate='sum'
        )
        self.gat_layer_2 = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GATv2Conv(embedding_size, embedding_size, num_heads=num_heads)
                for rel in rel_names
            },
            aggregate='sum'
        )

        self.output = nn.Linear(2 * embedding_size, embedding_size)

    def forward(self, g, h):
        # print(h)
        u = {'user': h['user']}
        i = {'item': h['item']}
        # user->item
        rsc = u
        dst = i
        h1 = self.gat_layer_1(g, (rsc, dst))
        h1['item'] = h1['item'].squeeze(2)

        # item->user
        rsc = i
        dst = u
        h2 = self.gat_layer_1(g, (rsc, dst))
        h2['user'] = h2['user'].squeeze(2)

        # item influence embedding: item->user
        rsc = h1
        dst = u
        item_influence_embedding = self.gat_layer_2(g, (rsc, dst))
        item_influence_embedding['user'] = item_influence_embedding['user'].squeeze(1)

        # social item embedding: user->user
        rsc = h2
        dst = u
        social_item_embedding = self.gat_layer_2(g, (rsc, dst))
        social_item_embedding['user'] = social_item_embedding['user'].squeeze(1)


        # print(item_influence_embedding, social_item_embedding)
        output = torch.cat([item_influence_embedding['user'], social_item_embedding['user']], dim=1)
        return self.output(output)

class SpectralAttentionLayer(nn.Module):
    def __init__(self, embedding_size=1, num_heads=1, kernel_nums=3):
        super(SpectralAttentionLayer, self).__init__()
        self.spec_gcn = dglnn.ChebConv(embedding_size, embedding_size, kernel_nums)
        self.att = dglnn.GATv2Conv(embedding_size, embedding_size, num_heads, allow_zero_in_degree=True)

    def forward(self, social_networks, h):
        u = h['user']
        laplacian_lambda_max = h['laplacian_lambda_max']
        # spectral gcn
        # print(u)
        h = self.spec_gcn(social_networks, u, laplacian_lambda_max)

        # attention
        h = self.att(social_networks, h)
        return h.squeeze(1)

class MutualisicLayer(nn.Module):
    def __init__(self, embedding_size=1):
        super(MutualisicLayer, self).__init__()
        self.consumption_mlp = nn.Linear(2 * embedding_size, embedding_size)
        self.social_mlp = nn.Linear(2 * embedding_size, embedding_size)

    def forward(self, raw_embed, consumption_pref, social_pref):
        # print(raw_embed)
        # print(consumption_pref)
        # print(social_pref)
        h_uP = self.consumption_mlp(torch.hstack([consumption_pref, raw_embed]))
        h_uS = self.social_mlp(torch.hstack([social_pref, raw_embed]))

        h_m = h_uP * h_uS

        atten_P = torch.softmax(h_uP, dim=1)
        h_mP = h_m * atten_P

        atten_S = torch.softmax(h_uS, dim=1)
        h_mS = h_m * atten_S

        h_miu_mP = torch.hstack([h_mP, h_uP])
        h_miu_mS = torch.hstack([h_mS, h_uS])
        return h_miu_mP, h_miu_mS

class PredictionLayer(nn.Module):
    def __init__(self, embedding_size=1):
        super(PredictionLayer, self).__init__()
        self.mutual_pref_mlp = nn.Linear(2 * embedding_size, embedding_size)
        self.mutual_social_mlp = nn.Linear(2 * embedding_size, embedding_size)


    def forward(self, **inputs):
        h_miu_mP = inputs['h_miu_mP']
        h_miu_mS = inputs['h_miu_mS']

        h_new_P = self.mutual_pref_mlp(h_miu_mP)
        h_new_S = self.mutual_social_mlp(h_miu_mS)

        return h_new_P, h_new_S

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, metatype, **embed):
        # h是从5.1节中对异构图的每种类型的边所计算的节点表示
        u_type, etype, v_type = metatype
        with graph.local_scope():
            if etype == 'rate':
                u_embed = embed['h_new_P']
                v_embed = embed['i_embed']
            elif etype == 'link':
                u_embed = embed['h_new_S']
                v_embed = embed['u_embed']
            else:
                raise ValueError("Wrong Etype!!")
            graph.nodes[u_type].data['h'] = u_embed
            graph.nodes[v_type].data['h'] = v_embed
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']
class BPRLoss(nn.Module):
    def __init__(self, balance_factor=1):
        self.balance_factor = balance_factor
        super(BPRLoss, self).__init__()

    def forward(self, output):
        pos_rate_score, neg_rate_score, pos_link_score, neg_link_score = output
        rate_loss = torch.sum(-torch.log(torch.sigmoid(pos_rate_score - neg_rate_score)))
        link_loss = torch.sum(-torch.log(torch.sigmoid(pos_link_score - neg_link_score)))
        return rate_loss * self.balance_factor, link_loss

class MutualRec(nn.Module):
    def __init__(self, config):
        super(MutualRec, self).__init__()
        embedding_size = eval(config['MODEL']['embedding_size'])
        num_heads = eval(config['MODEL']['num_heads'])
        num_kernels = eval(config['MODEL']['num_kernels'])
        num_nodes = {
            'user': eval(config['MODEL']['user_nums']),
            'item': eval(config['MODEL']['item_nums'])
        }
        self.embedding = dglnn.HeteroEmbedding(num_nodes, embedding_size)

        self.spatial_atten_layer = SpatialAttentionLayer_GAT(embedding_size, num_heads)
        self.spectral_atten_layer = SpectralAttentionLayer(embedding_size, num_heads, num_kernels)
        self.mutualistic_layer = MutualisicLayer(embedding_size)
        self.prediction_layer = PredictionLayer(embedding_size)

        self.predictor = HeteroDotProductPredictor()

    def generate_mask(self, train_g):
        self.rate_mask = train_g.to('cpu').adj(etype='rate').to_dense() == 0
        self.link_mask = train_g.to('cpu').adj(etype='link').to_dense() == 0

    def forward(self, g, train_pos_g, train_neg_g, social_networks, laplacian_lambda_max):
        # user_embed = torch.ones_like(g.nodes['user'].data['feat'])
        # item_embed = torch.ones_like(g.nodes['item'].data['feat'])
        user_item_embed = self.embedding({'user': g.nodes('user'), 'item': g.nodes('item')})
        # item_embed = self.embedding({'item': g.nodes('item')})
        # print(user_item_embed)
        user_embed = user_item_embed['user']
        item_embed = user_item_embed['item']
        user_social_embed = {
            'user': user_embed,
            'laplacian_lambda_max': laplacian_lambda_max
        }
        user_pref_embed = self.spatial_atten_layer(g, user_item_embed)
        user_social_embed = self.spectral_atten_layer(social_networks, user_social_embed)

        h_miu_mP, h_miu_mS = self.mutualistic_layer(user_embed, user_pref_embed, user_social_embed)

        h_new_P, h_new_S = self.prediction_layer(
            h_miu_mP = h_miu_mP,
            h_miu_mS = h_miu_mS,
        )
        pos_rate_score = self.predictor(train_pos_g, ('user', 'rate', 'item'), h_new_P=h_new_P, i_embed=item_embed)
        neg_rate_score = self.predictor(train_neg_g, ('user', 'rate', 'item'), h_new_P=h_new_P, i_embed=item_embed)

        pos_link_score = self.predictor(train_pos_g, ('user', 'link', 'user'), h_new_S=h_new_S, u_embed=user_embed)
        neg_link_score = self.predictor(train_neg_g, ('user', 'link', 'user'), h_new_S=h_new_S, u_embed=user_embed)
        return pos_rate_score, neg_rate_score, pos_link_score, neg_link_score

    def evaluate(self, g, test_pos_g, test_neg_g, social_networks, laplacian_lambda_max):
        # user_embed = torch.ones_like(g.nodes['user'].data['feat'])
        # item_embed = torch.ones_like(g.nodes['item'].data['feat'])
        user_item_embed = self.embedding({'user': g.nodes('user'), 'item': g.nodes('item')})
        # item_embed = self.embedding({'item': g.nodes('item')})
        # print(user_item_embed)
        user_embed = user_item_embed['user']
        item_embed = user_item_embed['item']
        user_social_embed = {
            'user': user_embed,
            'laplacian_lambda_max': laplacian_lambda_max
        }
        user_pref_embed = self.spatial_atten_layer(g, user_item_embed)
        user_social_embed = self.spectral_atten_layer(social_networks, user_social_embed)

        h_miu_mP, h_miu_mS = self.mutualistic_layer(user_embed, user_pref_embed, user_social_embed)

        h_new_P, h_new_S = self.prediction_layer(
            h_miu_mP = h_miu_mP,
            h_miu_mS = h_miu_mS,
        )
        pos_rate_score = self.predictor(test_pos_g, ('user', 'rate', 'item'), h_new_P=h_new_P, i_embed=item_embed)
        neg_rate_score = self.predictor(test_neg_g, ('user', 'rate', 'item'), h_new_P=h_new_P, i_embed=item_embed)

        pos_link_score = self.predictor(test_pos_g, ('user', 'link', 'user'), h_new_S=h_new_S, u_embed=user_embed)
        neg_link_score = self.predictor(test_neg_g, ('user', 'link', 'user'), h_new_S=h_new_S, u_embed=user_embed)

        rate_pred = torch.topk(torch.matmul(h_new_P, item_embed.t()).detach().cpu() * self.rate_mask, k=25, dim=1)[1].tolist()
        link_pred = torch.topk(torch.matmul(h_new_S, user_embed.t()).detach().cpu() * self.link_mask, k=25, dim=1)[1].tolist()

        return pos_rate_score, neg_rate_score, pos_link_score, neg_link_score, rate_pred, link_pred

def generate_pos_neg_g(rate_dict, link_dict, num_nodes_dict, mode='train'):
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
    pos_g = dgl.heterograph(pos_graph_data, num_nodes_dict)
    neg_g = dgl.heterograph(neg_graph_data, num_nodes_dict)
    return pos_g, neg_g

def get_pos_neg_edges(g, u, v, train_mask, test_mask, etype):
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

def prepare_debug_data():
    # u1 = torch.tensor([0, 0, 1, 1, 2, 3, 3, 4])
    # i = torch.tensor([0, 0, 1, 1, 2, 3, 3, 4])
    # u2 = torch.tensor([1, 2, 2, 3, 3, 4, 0, 0])
    # social_u = torch.cat([u1, u2])
    # social_v = torch.cat([u2, u1])
    # graph_data = {
    #     ('user', 'rate', 'item'): (u1, i),
    #     ('item', 'rated-by', 'user'): (i, u1),
    #     ('user', 'link', 'user'): (social_u, social_v)
    # }
    # g = dgl.heterograph(graph_data)

    from dataset import Epinions
    dataset = Epinions()
    g = dataset[0]

    u, i = g.edges(etype='rate')
    train_rate_mask = torch.tensor([0, 2, 4, 5, 7])
    test_rate_mask = torch.tensor([1, 3, 6])
    train_rate_mask = dataset.train_mask['rate']
    test_rate_mask = dataset.test_mask['rate']
    rate_dict = get_pos_neg_edges(g, u, i, train_rate_mask, test_rate_mask, etype='rate')

    u, i = g.edges(etype='link')
    train_link_mask = torch.tensor([0, 2, 4, 5, 7, 8, 10, 12, 13, 15])
    test_link_mask = torch.tensor([1, 3, 6, 9, 11, 14])
    train_link_mask = dataset.train_mask['link']
    test_link_mask = dataset.test_mask['link']
    link_dict = get_pos_neg_edges(g, u, i, train_link_mask, test_link_mask, etype='link')

    num_nodes_dict = {
        'user': g.number_of_nodes('user'),
        'item': g.number_of_nodes('item')
    }

    train_pos_g, train_neg_g = generate_pos_neg_g(rate_dict, link_dict, num_nodes_dict, 'train')
    test_pos_g, test_neg_g = generate_pos_neg_g(rate_dict, link_dict, num_nodes_dict, 'test')

    train_g = dgl.remove_edges(g, test_rate_mask, 'rate')
    train_g = dgl.remove_edges(train_g, test_rate_mask, 'rated-by')
    train_g = dgl.remove_edges(train_g, test_link_mask, 'link')

    social_networks = dgl.edge_type_subgraph(train_g, [('user', 'link', 'user')])

    laplacian_lambda_max = torch.tensor(dgl.laplacian_lambda_max(social_networks), dtype=torch.float32)
    return train_g, train_pos_g, train_neg_g, social_networks, laplacian_lambda_max, test_pos_g, test_neg_g

if __name__ == '__main__':

    train_g, train_pos_g, train_neg_g, social_networks, laplacian_lambda_max, test_pos_g, test_neg_g = prepare_debug_data()
    nodes_nums = {
        'user': train_g.num_nodes('user'),
        'item': train_g.num_nodes('item'),
    }

    model = MutualRec(embedding_size=128, num_nodes=nodes_nums)
    model.generate_mask(train_g)
    from weight_init import weight_init
    model.apply(weight_init)
    loss_func = BPRLoss()
    optimizer = torch.optim.Adam(lr=5e-3, params=model.parameters())
    model.to('cuda')
    train_g = train_g.to('cuda:0')
    train_pos_g = train_pos_g.to('cuda:0')
    train_neg_g = train_neg_g.to('cuda:0')
    test_pos_g = test_pos_g.to('cuda:0')
    test_neg_g = test_neg_g.to('cuda:0')
    social_networks = social_networks.to('cuda:0')
    laplacian_lambda_max = laplacian_lambda_max.to('cuda:0')

    for i in range(500):
        optimizer.zero_grad()
        # pos_rate_score, neg_rate_score, pos_link_score, neg_link_score = model(
        #     train_g, train_pos_g, train_neg_g, social_networks, laplacian_lambda_max)
        pos_rate_score, neg_rate_score, pos_link_score, neg_link_score = model(
            g=train_g,
            train_pos_g=train_pos_g,
            train_neg_g=train_neg_g,
            social_networks=social_networks,
            laplacian_lambda_max=laplacian_lambda_max
        )
        rate_loss, link_loss = loss_func(pos_rate_score, neg_rate_score, pos_link_score, neg_link_score)
        loss = rate_loss + link_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            output = model.evaluate(
                g=train_g,
                test_pos_g=test_pos_g,
                test_neg_g=test_neg_g,
                social_networks=social_networks,
                laplacian_lambda_max=laplacian_lambda_max
            )
        print(f'epoch:{i + 1}\tloss:{loss.item()}\trate:{rate_loss.item()}\tlink:{link_loss.item()}')