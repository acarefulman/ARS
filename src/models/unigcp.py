# coding: utf-8
# unigcp.py: Unified, Graph-enhanced, and Causality-aware Predictor for MMRec

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from open_clip import create_model_and_transforms
from utils.utils import build_sim, build_knn_normalized_graph
from common.abstract_recommender import GeneralRecommender


class UniCL_Embed(nn.Module):
    def __init__(self, clip_model_name='ViT-B-32', pretrained='laion2b_s34b_b79k'):
        super().__init__()
        self.model, _, _ = create_model_and_transforms(clip_model_name, pretrained=pretrained)
        self.model.eval()

    def forward(self, image_tensor, text_tensor):
        with torch.no_grad():
            image_embed = self.model.encode_image(image_tensor)
            text_embed = self.model.encode_text(text_tensor)
        return image_embed, text_embed


class SAGEAlign(nn.Module):
    def __init__(self, embed_dim, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, embeds, adj, initial_embed):
        h = embeds
        for _ in range(self.n_layers):
            h = torch.sparse.mm(adj, h)
            gate_val = self.gate(h)
            h = self.dropout(h * gate_val + initial_embed * (1 - gate_val))
        return h


class CPMS(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, image_embed, text_embed, fusion_embed):
        pred_all = self.predictor(fusion_embed)
        pred_no_img = self.predictor(text_embed)
        pred_no_txt = self.predictor(image_embed)

        delta_img = (pred_all - pred_no_img).norm(p=2, dim=1, keepdim=True)
        delta_txt = (pred_all - pred_no_txt).norm(p=2, dim=1, keepdim=True)

        weights = torch.cat([delta_img, delta_txt], dim=1)
        weights = F.softmax(weights, dim=1)
        fused = weights[:, 0:1] * image_embed + weights[:, 1:2] * text_embed
        return fused


class UNIGCP(GeneralRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.embedding_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.device = config['device']
        self.reg_weight = config['reg_weight']
        self.batch_size = config['batch_size']

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.clip_encoder = UniCL_Embed(config['clip_model_name'], config['pretrained'])
        self.text_proj = nn.Linear(512, self.embedding_dim)
        self.image_proj = nn.Linear(512, self.embedding_dim)

        self.image_adj = build_knn_normalized_graph(build_sim(dataset.image_feat), 20, norm_type='sym').to(self.device)
        self.text_adj = build_knn_normalized_graph(build_sim(dataset.text_feat), 20, norm_type='sym').to(self.device)

        self.graph_encoder = SAGEAlign(self.embedding_dim, self.n_layers)
        self.fusion_layer = CPMS(self.embedding_dim)

        self.R = dataset.inter_matrix(form='coo').astype('float32')
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype('float32')
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype('int64'))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def forward(self, image_tensor, text_tensor):
        image_feat, text_feat = self.clip_encoder(image_tensor, text_tensor)
        image_embed = self.image_proj(image_feat)
        text_embed = self.text_proj(text_feat)

        fusion_embed = (image_embed + text_embed) / 2

        image_graph_embed = self.graph_encoder(image_embed, self.image_adj, image_embed)
        text_graph_embed = self.graph_encoder(text_embed, self.text_adj, text_embed)
        fusion_graph_embed = self.graph_encoder(fusion_embed, self.image_adj, fusion_embed)

        final_item_embed = self.fusion_layer(image_graph_embed, text_graph_embed, fusion_graph_embed)
        final_user_embed = self.user_embedding.weight

        all_embeds = final_user_embed @ final_item_embed.T
        return all_embeds

    def predict(self, user_ids, item_ids, image_tensor, text_tensor):
        scores = self.forward(image_tensor, text_tensor)
        return scores[user_ids, item_ids]

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        emb_loss = self.reg_weight * (
                    users.norm(2).pow(2) + pos_items.norm(2).pow(2) + neg_items.norm(2).pow(2)) / 2 / self.batch_size
        return mf_loss, emb_loss, 0.0

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction
        image_tensor = self.dataset.image_tensor[pos_items].to(self.device)
        text_tensor = self.dataset.text_tensor[pos_items].to(self.device)

        all_embeds = self.forward(image_tensor, text_tensor)
        u = self.user_embedding.weight[users]
        pi = all_embeds[users, pos_items]
        ni = all_embeds[users, neg_items]

        mf_loss, emb_loss, reg_loss = self.bpr_loss(u, pi, ni)
        return mf_loss + emb_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        image_tensor = self.dataset.image_tensor.to(self.device)
        text_tensor = self.dataset.text_tensor.to(self.device)
        all_embeds = self.forward(image_tensor, text_tensor)
        return all_embeds[user]