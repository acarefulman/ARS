# coding: utf-8
# rongqing001@e.ntu.edu.sg
r"""
SMORE - Multi-modal Recommender System (Enhanced with Micro-Innovations)
Ablation: Disabled Attention-Driven Dynamic Graph Propagation
- Removed attention mechanism (att_image, att_text, att_fusion, att_dropout) in propagate
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, build_knn_normalized_graph


class SMORE_NO_ATTENTION(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SMORE_NO_ATTENTION, self).__init__(config, dataset)
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.image_knn_k = config['image_knn_k']
        self.text_knn_k = config['text_knn_k']
        self.dropout_rate = config['dropout_rate']
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, f'image_adj_{self.image_knn_k}_{self.sparse}.pt')
        text_adj_file = os.path.join(dataset_path, f'text_adj_{self.text_knn_k}_{self.sparse}.pt')

        self.norm_adj = self.get_adj_mat()
        self.R_sprse_mat = self.R
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.image_knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.text_knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        self.fusion_adj = self.max_pool_fusion()

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=0)

        self.query_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        )
        self.query_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        )

        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_f = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_fusion_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.image_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2))
        self.text_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2))
        self.fusion_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2))

        self.att_image = nn.Parameter(torch.Tensor(self.n_items, 1))
        self.att_text = nn.Parameter(torch.Tensor(self.n_items, 1))
        self.att_fusion = nn.Parameter(torch.Tensor(self.n_items, 1))
        nn.init.xavier_uniform_(self.att_image)
        nn.init.xavier_uniform_(self.att_text)
        nn.init.xavier_uniform_(self.att_fusion)
        self.att_dropout = nn.Dropout(p=0.2)

        self.alpha_v = nn.Parameter(torch.tensor(1.0))
        self.alpha_t = nn.Parameter(torch.tensor(1.0))
        self.alpha_f = nn.Parameter(torch.tensor(1.0))

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32).tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv).tocoo()
        norm_adj = norm_adj.tolil()
        self.R = norm_adj[:self.n_users, self.n_users:]
        return norm_adj.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        return torch.sparse_coo_tensor(indices, values, torch.Size(sparse_mx.shape))

    def spectrum_convolution(self, image_embeds, text_embeds):
        image_fft = torch.fft.rfft(image_embeds, dim=1, norm='ortho')
        text_fft = torch.fft.rfft(text_embeds, dim=1, norm='ortho')
        image_conv = torch.fft.irfft(image_fft * torch.view_as_complex(self.image_complex_weight),
                                     n=image_embeds.shape[1], dim=1, norm='ortho')
        text_conv = torch.fft.irfft(text_fft * torch.view_as_complex(self.text_complex_weight),
                                    n=text_embeds.shape[1], dim=1, norm='ortho')
        fusion_conv = torch.fft.irfft(text_fft * image_fft * torch.view_as_complex(self.fusion_complex_weight),
                                      n=text_embeds.shape[1], dim=1, norm='ortho')
        return image_conv, text_conv, fusion_conv

    def max_pool_fusion(self):
        image_adj, text_adj = self.image_original_adj.coalesce(), self.text_original_adj.coalesce()
        image_indices, text_indices = image_adj.indices(), text_adj.indices()
        image_values, text_values = image_adj.values(), text_adj.values()
        combined_indices = torch.cat((image_indices, text_indices), dim=1)
        combined_indices, unique_idx = torch.unique(combined_indices, dim=1, return_inverse=True)
        v_i = torch.full((combined_indices.size(1),), float('-inf')).to(self.device)
        v_t = torch.full((combined_indices.size(1),), float('-inf')).to(self.device)
        v_i[unique_idx[:image_indices.size(1)]] = image_values
        v_t[unique_idx[image_indices.size(1):]] = text_values
        v_all, _ = torch.max(torch.stack((v_i, v_t)), dim=0)
        return torch.sparse.FloatTensor(combined_indices, v_all, image_adj.size()).coalesce()

    def forward(self, adj, train=False):
        image_feats = self.image_trs(self.image_embedding.weight) if self.v_feat is not None else None
        text_feats = self.text_trs(self.text_embedding.weight) if self.t_feat is not None else None
        image_conv, text_conv, fusion_conv = self.spectrum_convolution(image_feats, text_feats)

        image_item_embeds = self.item_id_embedding.weight * self.gate_v(image_conv)
        text_item_embeds = self.item_id_embedding.weight * self.gate_t(text_conv)
        fusion_item_embeds = self.item_id_embedding.weight * self.gate_f(fusion_conv)

        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, self.item_id_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        for _ in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings + 0.5 * ego_embeddings
            all_embeddings.append(ego_embeddings)
        content_embeds = torch.stack(all_embeddings, dim=1).mean(dim=1)

        def propagate(modality_embeds, adj_matrix, att):
            for _ in range(self.n_layers):
                modality_embeds = torch.sparse.mm(adj_matrix, modality_embeds)  # Removed attention
            user_view = torch.sparse.mm(self.R, modality_embeds)
            return torch.cat([user_view, modality_embeds], dim=0)

        image_embeds = propagate(image_item_embeds, self.image_original_adj, self.att_image)
        text_embeds = propagate(text_item_embeds, self.text_original_adj, self.att_text)
        fusion_embeds = propagate(fusion_item_embeds, self.fusion_adj, self.att_fusion)

        fusion_att_v, fusion_att_t = self.query_v(fusion_embeds), self.query_t(fusion_embeds)
        agg_image_embeds = self.softmax(fusion_att_v) * image_embeds
        agg_text_embeds = self.softmax(fusion_att_t) * text_embeds

        image_prefer = self.dropout(self.gate_image_prefer(content_embeds))
        text_prefer = self.dropout(self.gate_text_prefer(content_embeds))
        fusion_prefer = self.dropout(self.gate_fusion_prefer(content_embeds))

        agg_image_embeds = agg_image_embeds * image_prefer
        agg_text_embeds = agg_text_embeds * text_prefer
        fusion_embeds = fusion_embeds * fusion_prefer

        modality_weights = self.softmax(torch.stack([self.alpha_v, self.alpha_t, self.alpha_f]))

        side_embeds = (modality_weights[0] * agg_image_embeds +
                       modality_weights[1] * agg_text_embeds +
                       modality_weights[2] * fusion_embeds)

        all_embeds = content_embeds + side_embeds
        return torch.split(all_embeds, [self.n_users, self.n_items], dim=0) if not train else (
            *torch.split(all_embeds, [self.n_users, self.n_items], dim=0), side_embeds, content_embeds)

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        emb_loss = self.reg_weight * (users.norm(2).pow(2) + pos_items.norm(2).pow(2) + neg_items.norm(2).pow(2)) / 2 / self.batch_size
        return mf_loss, emb_loss, 0.0

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos = torch.exp((view1 * view2).sum(dim=1) / temperature)
        ttl = torch.exp(view1 @ view2.T / temperature).sum(dim=1)
        return torch.mean(-torch.log(pos / ttl))

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction
        ua_e, ia_e, side_e, content_e = self.forward(self.norm_adj, train=True)
        u, pi, ni = ua_e[users], ia_e[pos_items], ia_e[neg_items]
        mf_loss, emb_loss, reg_loss = self.bpr_loss(u, pi, ni)
        su, si = torch.split(side_e, [self.n_users, self.n_items], dim=0)
        cu, ci = torch.split(content_e, [self.n_users, self.n_items], dim=0)
        cl = self.InfoNCE(si[pos_items], ci[pos_items], 0.2) + self.InfoNCE(su[users], cu[users], 0.2)
        return mf_loss + emb_loss + reg_loss + self.cl_loss * cl

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_e, i_e = self.forward(self.norm_adj)
        return u_e[user] @ i_e.T