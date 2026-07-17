import math
import os.path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, linalg

from src.common.abstract_recommender import GeneralRecommender


class ARS(GeneralRecommender):          # 类名改为 ARS
    def __init__(self, config, dataset):
        super(ARS, self).__init__(config, dataset)
        self.current_epoch = None
        self.embedding_dim = config['embedding_size']
        self.gcn_layers = config['gcn_layers']
        self.use_id = config['use_id']
        self.knn_k = config['knn_k']
        self.dropout = config['dropout']
        self.tau = config['tau']
        self.asg = config['asg']
        self.auige = config['auige']
        self.m_alpha = config['m_alpha']
        self.pe = config['pe']
        self.lamda = config['lamda']
        self.save_intermediate = config['save_intermediate']
        self.save_path = config['save_path']
        self.use_text = config['use_text']
        self.use_image = config['use_image']

        self.intermediate_path = os.path.abspath(os.path.join(self.save_path, config['model'], config['dataset']))
        if not os.path.exists(self.intermediate_path):
            os.makedirs(self.intermediate_path)

        self.n_nodes = self.n_users + self.n_items
        interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.ui_indices = torch.LongTensor(np.vstack((interaction_matrix.row, interaction_matrix.col))).to(self.device)
        self.base_adj = self.get_base_adj(self.ui_indices.clone())

        # init embeddings
        if self.use_id:
            self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
            self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
            nn.init.xavier_uniform_(self.user_embedding.weight)
            nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # 构建模态相关邻接和特征投影
        u_v_adj, i_v_adj, u_t_adj, i_t_adj = None, None, None, None
        if self.v_feat is not None and self.use_image:
            self.v_feat_i = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.v_feat_u = self.cal_user_embedding_mean(self.v_feat)
            i_v_adj = self.get_knn_adj(self.v_feat_i.weight)
            u_v_adj = self.get_knn_adj(self.v_feat_u)
        if self.t_feat is not None and self.use_text:
            self.t_feat_i = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.t_feat_u = self.cal_user_embedding_mean(self.t_feat)
            i_t_adj = self.get_knn_adj(self.t_feat_i.weight)
            u_t_adj = self.get_knn_adj(self.t_feat_u)
        if self.use_text and self.use_image:
            self.base_ii = i_v_adj + i_t_adj
            self.base_uu = u_v_adj + u_t_adj
        elif self.use_text:
            self.base_ii = i_t_adj
            self.base_uu = u_t_adj
        elif self.use_image:
            self.base_ii = i_v_adj
            self.base_uu = u_v_adj
        else:
            raise ValueError("At least one of use_text or use_image must be True.")

        self.i_pe = PositionalEncoding(self.embedding_dim, self.n_items, self.device)
        self.u_pe = PositionalEncoding(self.embedding_dim, self.n_users, self.device)
        if self.use_text or self.use_image:
            self.i_edge_predictor = MLP_view(self.base_ii, self.embedding_dim, self.device)
            self.u_edge_predictor = MLP_view(self.base_uu, self.embedding_dim, self.device)
        if self.v_feat is not None and self.use_image:
            self.v_mu = MLP(self.v_feat.size(-1), self.embedding_dim)
        if self.t_feat is not None and self.use_text:
            self.t_mu = MLP(self.t_feat.size(-1), self.embedding_dim)

        # ---------- ARS 核心：注意力评分器（两层MLP）----------
        # 共享评分器，输入维度为 embedding_dim，输出标量
        self.scorer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),  # 隐藏层维度可调，这里设为与嵌入相同
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1)
        )
        # 可选：显式初始化
        for layer in self.scorer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    # ---------- 以下方法保持原样 ----------
    def cal_user_embedding_mean(self, embeddings):
        rows = self.ui_indices[0]
        cols = self.ui_indices[1]
        item_embeddings = embeddings[cols]
        user_embedding_sum = torch.zeros((self.n_users, embeddings.size(-1)), device=self.device)
        user_interaction_count = torch.zeros(self.n_users, device=self.device)
        user_embedding_sum.index_add_(0, rows, item_embeddings)
        user_interaction_count.index_add_(0, rows, torch.ones_like(rows, dtype=torch.float32))
        user_embedding_mean = user_embedding_sum / user_interaction_count.unsqueeze(1)
        user_embedding_mean = torch.nan_to_num(user_embedding_mean, nan=0.0, posinf=0.0, neginf=0.0)
        return user_embedding_mean

    def get_base_adj(self, ui_indices):
        adj_size = torch.Size((self.n_nodes, self.n_nodes))
        ui_indices[1] += self.n_users
        ui_graph = torch.sparse_coo_tensor(ui_indices, torch.ones_like(self.ui_indices[0], dtype=torch.float32),
                                           adj_size, device=self.device)
        iu_graph = ui_graph.T
        base_adj = ui_graph + iu_graph
        return base_adj

    def get_aug_adj_mat(self, base_adj, uu_graph, ii_graph):
        if uu_graph is None and ii_graph is None:
            return base_adj
        adj_size = torch.Size((self.n_nodes, self.n_nodes))
        uu_graph = torch.sparse_coo_tensor(uu_graph._indices(),
                                           uu_graph._values(), adj_size, device=self.device)
        ii_graph = torch.sparse_coo_tensor(ii_graph._indices() + self.n_users,
                                           ii_graph._values(), adj_size, device=self.device)
        aug_adj = uu_graph + base_adj + ii_graph
        return aug_adj

    def get_knn_adj(self, embeddings):
        context_norm = embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        knn_val, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        indices0 = torch.arange(knn_ind.size(0)).unsqueeze(1).expand(-1, self.knn_k).to(self.device)
        indices = torch.stack([indices0.flatten(), knn_ind.flatten()], dim=0)
        adj = torch.sparse_coo_tensor(indices, knn_val.flatten().squeeze(), sim.size())
        return adj

    def cal_norm_laplacian(self, adj):
        indices = adj._indices()
        values = adj._values()
        row = indices[0]
        col = indices[1]
        rowsum = torch.sparse.sum(adj, dim=-1).to_dense()
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0)
        row_inv_sqrt = d_inv_sqrt[row]
        col_inv_sqrt = d_inv_sqrt[col]
        values = values * row_inv_sqrt * col_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj.shape)

    def sample_adj(self, adj, dropout):
        edge_value = adj._values()
        edge_value[torch.isnan(edge_value)] = 0.
        degree_len = int(edge_value.size(0) * (1. - dropout))
        degree_idx = torch.multinomial(edge_value, degree_len)
        keep_indices = adj._indices()[:, degree_idx]
        new_adj = torch.sparse_coo_tensor(keep_indices, edge_value[degree_idx], adj.shape)
        return self.cal_norm_laplacian(new_adj)

    # ---------- 修改：返回所有层堆叠，不再做mean ----------
    def simple_gcn(self, ego_embeddings, norm_adj):
        """返回所有层（含初始层）的堆叠，shape: [n_nodes, layers+1, dim]"""
        all_embeddings = [ego_embeddings]
        for i in range(self.gcn_layers):
            side_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        # 堆叠所有层，不再平均
        all_embeddings = torch.stack(all_embeddings, dim=1)  # [n_nodes, L+1, dim]
        return all_embeddings

    # ---------- forward 中使用 ARS 聚合 ----------
    def forward(self, training):
        # 获取模态特征（经过投影和位置编码）
        if self.pe:
            if self.use_text and self.use_image:
                i_v_mu = self.i_pe(self.v_mu(self.v_feat_i.weight))
                u_v_mu = self.u_pe(self.v_mu(self.v_feat_u))
                i_t_mu = self.i_pe(self.t_mu(self.t_feat_i.weight))
                u_t_mu = self.u_pe(self.t_mu(self.t_feat_u))
            elif self.use_text:
                i_t_mu = self.i_pe(self.t_mu(self.t_feat_i.weight))
                u_t_mu = self.u_pe(self.t_mu(self.t_feat_u))
                i_v_mu, u_v_mu = None, None
            elif self.use_image:
                i_v_mu = self.i_pe(self.v_mu(self.v_feat_i.weight))
                u_v_mu = self.u_pe(self.v_mu(self.v_feat_u))
                i_t_mu, u_t_mu = None, None
            else:
                i_v_mu, u_v_mu = None, None
                i_t_mu, u_t_mu = None, None
        else:
            i_v_mu = self.v_mu(self.v_feat_i.weight) if self.use_image else None
            u_v_mu = self.v_mu(self.v_feat_u) if self.use_image else None
            i_t_mu = self.t_mu(self.t_feat_i.weight) if self.use_text else None
            u_t_mu = self.t_mu(self.t_feat_u) if self.use_text else None

        # 构造GCN的初始输入（融合模态或仅ID）
        if self.use_text and self.use_image:
            u_embedding = self.m_alpha * u_t_mu + (1 - self.m_alpha) * u_v_mu
            i_embedding = self.m_alpha * i_t_mu + (1 - self.m_alpha) * i_v_mu
        elif self.use_text:
            u_embedding = u_t_mu
            i_embedding = i_t_mu
        elif self.use_image:
            u_embedding = u_v_mu
            i_embedding = i_v_mu
        else:
            u_embedding = self.u_pe(torch.zeros((self.n_users, self.embedding_dim), device=self.device))
            i_embedding = self.i_pe(torch.zeros((self.n_items, self.embedding_dim), device=self.device))
        if self.use_id:
            u_embedding = u_embedding + self.user_embedding.weight
            i_embedding = i_embedding + self.item_id_embedding.weight

        # 构造多模态图邻接
        if self.auige and (self.use_text or self.use_image):
            if self.asg:
                if self.training:
                    if self.use_text and self.use_image:
                        ii_adj = self.i_edge_predictor(i_t_mu, i_v_mu)
                        uu_adj = self.u_edge_predictor(u_t_mu, u_v_mu)
                    elif self.use_text:
                        ii_adj = self.i_edge_predictor(i_t_mu, i_t_mu)
                        uu_adj = self.u_edge_predictor(u_t_mu, u_t_mu)
                    elif self.use_image:
                        ii_adj = self.i_edge_predictor(i_v_mu, i_v_mu)
                        uu_adj = self.u_edge_predictor(u_v_mu, u_v_mu)
                    else:
                        ii_adj, uu_adj = None, None
                else:
                    ii_adj = self.get_knn_adj(i_embedding)
                    uu_adj = self.get_knn_adj(u_embedding)
                adj = self.get_aug_adj_mat(self.base_adj, uu_adj, ii_adj).coalesce()
            else:
                adj = self.get_aug_adj_mat(self.base_adj, self.base_uu, self.base_ii).coalesce()
        else:
            adj = self.base_adj

        # LightGCN 传播
        if training and self.dropout > 0.:
            norm_adj = self.sample_adj(adj, self.dropout)
        else:
            norm_adj = self.cal_norm_laplacian(adj)

        all_embeddings = torch.cat([u_embedding, i_embedding], dim=0)          # [n_nodes, dim]
        all_layer_embeddings = self.simple_gcn(all_embeddings, norm_adj)       # [n_nodes, L+1, dim]

        # 分离用户和物品的层输出
        u_layer_embeds = all_layer_embeddings[:self.n_users]   # [n_users, L+1, dim]
        i_layer_embeds = all_layer_embeddings[self.n_users:]   # [n_items, L+1, dim]

        # ---------- ARS 聚合：构建候选集并学习选择权重 ----------
        # 用户候选：所有层 + 模态特征（如果存在）
        user_candidates = [u_layer_embeds]  # 列表，每个元素 [n_users, dim]
        # 将层输出展平为 [n_users, num_layers, dim] -> 我们将其视为多个候选，先拆分
        # 更简单：将层输出和模态特征在第二维拼接
        # 准备模态候选
        modal_candidates_u = []
        if u_t_mu is not None:
            modal_candidates_u.append(u_t_mu)
        if u_v_mu is not None:
            modal_candidates_u.append(u_v_mu)
        # 如果没有任何模态，则只用层
        if len(modal_candidates_u) == 0:
            # 直接使用层输出平均？但这里我们只取层输出
            # 为了统一，我们将层输出视为唯一候选，直接加权（其实等价于普通加权）
            # 但为了符合论文，最好有至少两个候选，但如果没有模态，我们仍可以只对层加权
            pass

        # 将层输出和模态候选拼接成 [n_users, num_candidates, dim]
        # 注意层输出有 L+1 个，每个形状 [n_users, dim]
        # 将 u_layer_embeds 按第二维拆分并添加到列表
        layer_list = [u_layer_embeds[:, i, :] for i in range(u_layer_embeds.size(1))]
        candidates_u = layer_list + modal_candidates_u
        if len(candidates_u) == 0:
            # 极少数情况，但不会发生，因为至少有层
            # 若没有候选，则用零
            final_u = torch.zeros((self.n_users, self.embedding_dim), device=self.device)
        else:
            candidates_u = torch.stack(candidates_u, dim=1)   # [n_users, num_candidates, dim]
            # 计算每个候选的得分
            scores_u = self.scorer(candidates_u).squeeze(-1)  # [n_users, num_candidates]
            weights_u = F.softmax(scores_u, dim=1)            # [n_users, num_candidates]
            final_u = torch.sum(weights_u.unsqueeze(-1) * candidates_u, dim=1)  # [n_users, dim]

        # 同理物品
        layer_list_i = [i_layer_embeds[:, i, :] for i in range(i_layer_embeds.size(1))]
        modal_candidates_i = []
        if i_t_mu is not None:
            modal_candidates_i.append(i_t_mu)
        if i_v_mu is not None:
            modal_candidates_i.append(i_v_mu)
        candidates_i = layer_list_i + modal_candidates_i
        if len(candidates_i) == 0:
            final_i = torch.zeros((self.n_items, self.embedding_dim), device=self.device)
        else:
            candidates_i = torch.stack(candidates_i, dim=1)
            scores_i = self.scorer(candidates_i).squeeze(-1)
            weights_i = F.softmax(scores_i, dim=1)
            final_i = torch.sum(weights_i.unsqueeze(-1) * candidates_i, dim=1)

        if training:
            # 返回聚合后的用户/物品嵌入以及模态特征（用于辅助损失）
            return final_u, final_i, u_t_mu, i_t_mu, u_v_mu, i_v_mu
        else:
            return final_u, final_i, i_t_mu, i_v_mu

    # ---------- loss 和预测保持不变 ----------
    def sl_loss(self, users, pos_items, neg_items):
        pos_scores = F.cosine_similarity(users, pos_items)
        neg_scores = F.cosine_similarity(users.unsqueeze(1), neg_items, dim=2)
        d = neg_scores - pos_scores.unsqueeze(1)
        loss = torch.logsumexp(d / self.tau, dim=1).mean()
        return loss

    def infonce_loss(self, emb1, emb2):
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)
        scores = torch.exp(torch.matmul(emb1, emb2.T) / self.tau)
        pos_sim = scores.diag()
        loss = -torch.log(pos_sim / torch.sum(scores, dim=1)).mean()
        return loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_g_embeddings, i_g_embeddings, u_t_mu, i_t_mu, u_v_mu, i_v_mu = self.forward(True)
        # 模态间对齐损失
        m_loss = 0.
        if self.use_text and self.use_image:
            u_m_loss = self.infonce_loss(u_t_mu[users], u_v_mu[users])
            i_m_loss = self.infonce_loss(i_t_mu[pos_items], i_v_mu[pos_items])
            m_loss = u_m_loss + i_m_loss

        loss_rec = self.sl_loss(u_g_embeddings[users], i_g_embeddings[pos_items], i_g_embeddings[neg_items])
        return loss_rec + self.lamda * m_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_g_embeddings, i_g_embeddings, i_t_mu, i_v_mu = self.forward(False)
        if self.save_intermediate:
            torch.save(u_g_embeddings, os.path.join(self.intermediate_path, f'u_embeddings_{self.current_epoch}.pt'))
            torch.save(i_g_embeddings, os.path.join(self.intermediate_path, f'i_embeddings_{self.current_epoch}.pt'))
        u_embeddings = u_g_embeddings[user]
        scores = torch.matmul(u_embeddings, i_g_embeddings.transpose(0, 1))
        return scores

    def pre_epoch_processing(self, epoch_idx):
        self.current_epoch = epoch_idx


# ---------- 辅助模块（保持不变） ----------
class MLP_view(nn.Module):
    def __init__(self, adj, in_dim, device):
        super(MLP_view, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU())
        self.edge_index = torch.clone(adj._indices())
        self.edge_val = torch.clone(adj._values())
        self.adj_size = adj.size()
        self.device = device

    def forward(self, Eu, Ev):
        Xu = self.fc1(Eu)
        Xv = self.fc2(Ev)
        src, dst = self.edge_index[0], self.edge_index[1]
        x_u, x_i = Xu[src], Xv[dst]
        edge_logits = torch.mul(x_u, x_i).sum(1).squeeze()
        edge_val = edge_logits
        Ag = self.edge_val * torch.sigmoid(edge_val)
        return torch.sparse_coo_tensor(self.edge_index, Ag, self.adj_size, device=self.device)


class PositionalEncoding(nn.Module):
    def __init__(self, pos_dim, max_len, device):
        super(PositionalEncoding, self).__init__()
        self.pos_dim = pos_dim
        self.max_len = max_len
        self.device = device
        self.pe = torch.zeros(max_len, pos_dim, device=self.device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pos_dim, 2).float() * (-math.log(10000.0) / pos_dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        x = x + self.pe
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1, activation="tanh", layer_norm=True):
        super(MLP, self).__init__()
        self.fcs = nn.ModuleList()
        if activation == 'tanh':
            activation_layer = nn.Tanh()
        elif activation == 'siLu':
            activation_layer = nn.SiLU()
        elif activation == "sigmoid":
            activation_layer = nn.Sigmoid()
        elif activation == "softmax":
            activation_layer = nn.Softmax(dim=-1)
        elif activation == "relu":
            activation_layer = nn.ReLU()
        else:
            activation_layer = None

        if input_dim <= output_dim:
            for i in range(num_layers):
                self.fcs.append(nn.Linear(input_dim, output_dim))
                if activation_layer is not None:
                    self.fcs.append(activation_layer)
                if layer_norm:
                    self.fcs.append(nn.LayerNorm(output_dim))
        else:
            step_ratio = (output_dim / input_dim) ** (1 / num_layers)
            dims = [input_dim]
            for i in range(1, num_layers):
                current_dim = input_dim * (step_ratio ** i)
                hidden_dim = self.next_power_of_two(current_dim)
                self.fcs.append(nn.Linear(dims[-1], hidden_dim))
                if activation_layer is not None:
                    self.fcs.append(activation_layer)
                if layer_norm:
                    self.fcs.append(nn.LayerNorm(hidden_dim))
                dims.append(hidden_dim)
            self.fcs.append(nn.Linear(dims[-1], output_dim))
            if activation_layer is not None:
                self.fcs.append(activation_layer)
            if layer_norm:
                self.fcs.append(nn.LayerNorm(output_dim))

    def forward(self, x):
        for layer in self.fcs:
            x = layer(x)
        return x

    def next_power_of_two(self, n):
        return 2 ** math.ceil(math.log2(n))