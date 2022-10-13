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
import torch
import torch.nn as nn
import numpy as np
import random

class SpatialAttentionLayer_GAT(nn.Module):
    def __init__(self, embedding_size=1, num_heads=1, rel_names=None):
        super(SpatialAttentionLayer_GAT, self).__init__()
        if rel_names is None:
            rel_names = ['rate', 'rated', 'friend']
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

        self.att = dglnn.GATv2Conv(embedding_size, embedding_size, num_heads)

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

        item_embed = inputs['item_embed']
        user_embed = inputs['user_embed']

        r_hat = torch.matmul(h_new_P, item_embed.t())
        s_hat = torch.matmul(h_new_S, user_embed.t())

        return r_hat, s_hat

class MutualRec(nn.Module):
    def __init__(self, embedding_size=1, num_heads=1, kernel_nums=3, n_nums=None):
        super(MutualRec, self).__init__()
        if n_nums is None:
            n_nums = {'user': 5, 'item': 4}
        self.embedding = dglnn.HeteroEmbedding(n_nums, embedding_size)

        self.spatial_atten_layer = SpatialAttentionLayer_GAT(embedding_size, num_heads)
        self.spectral_atten_layer = SpectralAttentionLayer(embedding_size, num_heads, kernel_nums)
        self.mutualistic_layer = MutualisicLayer(embedding_size)
        self.prediction_layer = PredictionLayer(embedding_size)


    def forward(self, g, social_networks, laplacian_lambda_max):
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

        r_hat, s_hat = self.prediction_layer(
            h_miu_mP = h_miu_mP,
            h_miu_mS = h_miu_mS,
            user_embed = user_embed,
            item_embed = item_embed
        )
        return r_hat, s_hat

class MutualRecLoss(nn.Module):
    def __init__(self):
        super(MutualRecLoss, self).__init__()

    def forward(self, **inputs):
        rate_pred = inputs['rate_pred']
        link_pred = inputs['link_pred']

        pos_u = inputs['pos_edges']['pos_u']
        pos_i = inputs['pos_edges']['pos_i']
        pos_u1 = inputs['pos_edges']['pos_u1']
        pos_u2 = inputs['pos_edges']['pos_u2']

        neg_u = inputs['neg_edges']['neg_u']
        neg_i = inputs['neg_edges']['neg_i']
        neg_u1 = inputs['neg_edges']['neg_u1']
        neg_u2 = inputs['neg_edges']['neg_u2']

        pos_rate = rate_pred[pos_u, pos_i]
        pos_link = link_pred[pos_u1, pos_u2]

        neg_rate = rate_pred[neg_u, neg_i]
        neg_link = link_pred[neg_u1, neg_u2]

        loss = torch.sum(-torch.log(torch.sigmoid(pos_rate - neg_rate))) - torch.sum(torch.log(torch.sigmoid(pos_link - neg_link)))
        return loss

def generate_neg_edges(g, neg_sampler):
    neg_edges = neg_sampler(g, eids={'friend': torch.arange(g.number_of_edges('friend')),
                         'rate': torch.arange(g.number_of_edges('rate'))})

    neg_rate_u, neg_rate_i = neg_edges[('user', 'rate', 'item')]
    neg_link_u1, neg_link_u2 = neg_edges[('user', 'friend', 'user')]


    neg_edges = {
        'neg_u': neg_rate_u,
        'neg_i': neg_rate_i,
        'neg_u1': neg_link_u1,
        'neg_u2': neg_link_u2,
    }
    return neg_edges

if __name__ == '__main__':
    u1 = torch.tensor([0, 0, 1, 2, 3, 3, 4])
    i = torch.tensor([0, 1, 1, 2, 2, 3, 3])
    u2 = torch.tensor([1, 2, 3, 4, 0, 2, 0])
    social_u = torch.cat([u1, u2])
    social_v = torch.cat([u2, u1])
    graph_data = {
        ('user', 'rate', 'item'): (u1, i),
        ('item', 'rated', 'user'): (i, u1),
        ('user', 'friend', 'user'): (social_u, social_v)
    }
    g = dgl.heterograph(graph_data)

    social_networks = dgl.edge_type_subgraph(g, [('user', 'friend', 'user')])

    laplacian_lambda_max = torch.tensor(dgl.laplacian_lambda_max(social_networks), dtype=torch.float32)

    rating_gt = torch.sparse_coo_tensor(torch.vstack([u1, i]), torch.ones_like(u1), (5, 4), dtype=torch.float32).to_dense()
    link_gt = torch.sparse_coo_tensor(torch.vstack([social_u, social_v]), torch.ones_like(social_u), (5, 5), dtype=torch.float32).to_dense()

    neg_sampler = dgl.dataloading.negative_sampler.GlobalUniform(k=1, exclude_self_loops=True, replace=True)

    pos_rate_u, pos_rate_i = g.edges(etype='rate')
    pos_link_u1, pos_link_u2 = g.edges(etype='friend')
    pos_edges = {
        'pos_u': pos_rate_u,
        'pos_i': pos_rate_i,
        'pos_u1': pos_link_u1,
        'pos_u2': pos_link_u2,
    }

    model = MutualRec(embedding_size=10)
    loss_func = MutualRecLoss()
    optimizer = torch.optim.Adam(lr=1e-3, params=model.parameters())
    for i in range(150):
        neg_edges = generate_neg_edges(g, neg_sampler)
        optimizer.zero_grad()
        rate_pred, link_pred = model(g, social_networks, laplacian_lambda_max)
        loss = loss_func(
            rate_pred=rate_pred,
            link_pred=link_pred,
            pos_edges=pos_edges,
            neg_edges=neg_edges
        )
        loss.backward()
        optimizer.step()
        print(f'epoch: {i}\tloss: {loss.item()}')