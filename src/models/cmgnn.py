# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import os
import torch.optim as optim
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product

class CausalGraph:
    def __init__(self, V, path, multimodal_features=None, config=None):
        self.v = list(V)
        self.set_v = set(V)
        self.multimodal_features = multimodal_features or {}
        self.fn = {v: set() for v in V}
        self.sn = {v: set() for v in V}
        self.on = {v: set() for v in V}
        self.p = set(map(tuple, map(sorted, path)))
        for v1, v2 in path:
            self.fn[v1].add(v2)
            self.fn[v2].add(v1)
            self.p.add(tuple(sorted((v1, v2))))
        self.config = config or {}
        self.n_users = sum(1 for v in V if v.startswith('user_'))
        self.n_items = sum(1 for v in V if v.startswith('item_'))
        self.n_nodes = len(V)
        self.adj_matrix = self.build_adj_matrix()
        self.norm_adj = self.get_norm_adj_mat()
        self.sub_graph = None
        self.mm_adj = None

    def build_adj_matrix(self):
        num_nodes = len(self.v)
        adj_matrix = torch.zeros((num_nodes, num_nodes))
        for v1, v2 in self.p:
            i, j = self.v.index(v1), self.v.index(v2)
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
        return adj_matrix

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        for v1, v2 in self.p:
            i, j = self.v.index(v1), self.v.index(v2)
            A[i, j] = 1
            A[j, i] = 1
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def get_knn_adj_mat(self, mm_embeddings, device):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.config.get('knn_k', 10), dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0]).to(device)
        indices0 = torch.unsqueeze(indices0, 1).expand(-1, self.config.get('knn_k', 10))
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def categorize_neighbors(self, target_node):
        if target_node not in self.set_v:
            return
        one_hop_neighbors = self.fn[target_node]
        two_hop_neighbors = set()
        for neighbor in one_hop_neighbors:
            two_hop_neighbors |= self.fn[neighbor]
        two_hop_neighbors -= one_hop_neighbors
        two_hop_neighbors.discard(target_node)
        out_of_neighborhood = self.set_v - (one_hop_neighbors | two_hop_neighbors | {target_node})
        self.sn[target_node] = two_hop_neighbors
        self.on[target_node] = out_of_neighborhood
        return target_node, one_hop_neighbors, two_hop_neighbors, out_of_neighborhood

    def plot(self):
        G = nx.Graph()
        G.add_nodes_from(self.v)
        G.add_edges_from(self.p)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=200, font_size=10, font_weight='bold', node_color="lightblue", edge_color="grey")
        plt.savefig('causal.png')
        plt.show()

class CMGNN(nn.Module):
    def __init__(self, config, train_data=None, v_feat=None, t_feat=None, device='cuda'):
        super(CMGNN, self).__init__()

        # 安全获取 config 参数的辅助函数，适配 Config 对象
        def get_config_value(key, default=None):
            try:
                # 尝试属性访问
                return getattr(config, key, default)
            except AttributeError:
                # 如果属性访问失败，返回默认值
                return default

        # 从 config 中获取 n_users 和 n_items，强制转换为整数
        self.n_users = int(get_config_value('n_users', 1000))
        self.n_items = int(get_config_value('n_items', 1000))
        self.embedding_dim = int(get_config_value('embedding_size', 64))
        self.feat_embed_dim = int(get_config_value('feat_embed_dim', 64))
        self.n_mm_layers = int(get_config_value('n_mm_layers', 1))
        self.n_ui_layers = int(get_config_value('n_ui_layers', 2))
        self.mm_image_weight = float(get_config_value('mm_image_weight', 0.1))

        # 处理 reg_weight 和 dropout 可能是列表的情况，取第一个元素
        reg_weight = get_config_value('reg_weight', 0.05)
        if isinstance(reg_weight, (list, tuple)) and len(reg_weight) > 0:
            reg_weight = float(reg_weight[0])
        self.reg_weight = reg_weight

        dropout_rate = get_config_value('dropout', 0.2)
        if isinstance(dropout_rate, (list, tuple)) and len(dropout_rate) > 0:
            dropout_rate = float(dropout_rate[0])
        self.dropout = nn.Dropout(dropout_rate)

        self.device = get_config_value('device', device)
        self.config = config

        # 初始化嵌入层，使用正确的 n_users 和 n_items
        self.user_text = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_image = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_image.weight)
        nn.init.xavier_uniform_(self.user_text.weight)

        self.v_feat = v_feat
        self.t_feat = t_feat

        # 尝试从 train_data 获取 CausalGraph
        self.graph = None
        if train_data is not None:
            if hasattr(train_data, 'dataset') and hasattr(train_data.dataset, 'graph'):
                self.graph = train_data.dataset.graph
            elif hasattr(train_data, 'graph') and isinstance(train_data.graph, CausalGraph):
                self.graph = train_data.graph
                print(f"Debug: graph from train_data = {self.graph}")
            elif isinstance(train_data, CausalGraph):
                self.graph = train_data

        mm_adj_file = os.path.join(
            get_config_value('data_path', './data'),
            get_config_value('dataset', 'default_dataset'),
            f'mm_adj_freedomdsp_{get_config_value("knn_k", 10)}_{int(10 * self.mm_image_weight)}.pt'
        )
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file).to(self.device)
        else:
            image_adj = None
            text_adj = None
            if v_feat is not None:
                self.image_embedding = nn.Embedding.from_pretrained(v_feat, freeze=False)
                self.image_trs = nn.Linear(v_feat.shape[1], self.feat_embed_dim)
                _, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach(), self.device)
            if t_feat is not None:
                self.text_embedding = nn.Embedding.from_pretrained(t_feat, freeze=False)
                self.text_trs = nn.Linear(t_feat.shape[1], self.feat_embed_dim)
                _, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach(), self.device)
            if v_feat is not None and t_feat is not None and image_adj is not None and text_adj is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj, image_adj
            elif v_feat is not None and image_adj is not None:
                self.mm_adj = image_adj
            elif t_feat is not None and text_adj is not None:
                self.mm_adj = text_adj
            else:
                self.mm_adj = None
            if self.mm_adj is not None:
                torch.save(self.mm_adj, mm_adj_file)

    def get_knn_adj_mat(self, mm_embeddings, device):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.config.get('knn_k', 10), dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0]).to(device)
        indices0 = torch.unsqueeze(indices0, 1).expand(-1, self.config.get('knn_k', 10))
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def forward(self, adj, graph):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        else:
            image_feats = torch.zeros(self.n_items, self.feat_embed_dim).to(self.device)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
        else:
            text_feats = torch.zeros(self.n_items, self.feat_embed_dim).to(self.device)

        image_feats = F.normalize(image_feats, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)
        user_embeds = torch.cat([self.user_image.weight, self.user_text.weight], dim=1)
        item_embeds = torch.cat([image_feats, text_feats], dim=1)

        h = item_embeds
        for i in range(self.n_mm_layers):
            h = torch.sparse.mm(self.mm_adj, h) if self.mm_adj is not None else h

        ego_embeddings = torch.cat((user_embeds, item_embeds), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        return mf_loss

    def InfoNCE(self, view1, view2, temperature=0.2):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        if self.graph is None:
            raise ValueError("Graph must be provided or set during model initialization")
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        ua_embeddings, ia_embeddings = self.forward(self.graph.norm_adj, self.graph)
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]
        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        cl_loss = (self.InfoNCE(self.dropout(u_g_embeddings), self.dropout(u_g_embeddings), 0.2) +
                   self.InfoNCE(self.dropout(pos_i_g_embeddings), self.dropout(pos_i_g_embeddings), 0.2)) / 2
        return batch_mf_loss + self.reg_weight * cl_loss

    def full_sort_predict(self, interaction, graph):
        user = interaction[0]
        restore_user_e, restore_item_e = self.forward(graph.norm_adj, graph)
        u_embeddings = restore_user_e[user]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    def pre_epoch_processing(self):
        # 实现预处理逻辑，例如初始化 epoch 状态
        pass

class NCM:
    def __init__(self, graph, target_node, learning_rate, h_size, h_layers, data, multimodal_features, config):
        self.graph = graph
        self.h_size = h_size
        self.h_layers = h_layers
        self.learning_rate = learning_rate
        self.target_node = target_node
        self.config = config
        self.model = CMGNN(config, train_data=graph, v_feat=multimodal_features.get('v_feat'), t_feat=multimodal_features.get('t_feat'), device=config.get('device', 'cuda'))
        self.states = {graph.target_node: torch.tensor([data.loc[graph.target_node, 'rating']], dtype=torch.float32) if target_node in data.index else torch.tensor([0.0], dtype=torch.float32)}
        self.u_i = {
            v: torch.cat([multimodal_features[v]['text'], multimodal_features[v]['image']], dim=-1) if v in multimodal_features else torch.zeros(self.h_size)
            for v in graph.one_hop_neighbors | graph.two_hop_neighbors
        }
        self.u_ij = self.u_i.copy()
        self.u = torch.cat([self.states[graph.target_node]] + [self.u_i.get(v, torch.zeros(self.h_size)) for v in graph.one_hop_neighbors | graph.two_hop_neighbors], dim=0)

    def add_gaussian_noise(self, tensor, mean=0.0, std=0.01):
        noise = torch.randn(tensor.size()) * std + mean
        return torch.clamp(tensor + noise, 0, 1)

    def ncm_forward(self, node_features, adj_matrix, add_noise=False):
        if add_noise:
            for k in self.u_i:
                self.u_i[k] = self.add_gaussian_noise(self.u_i[k])
            for k in self.u_ij:
                self.u_ij[k] = self.add_gaussian_noise(self.u_ij[k])
            self.u = torch.cat([self.states[self.graph.target_node]] + [self.u_i.get(v, torch.zeros(self.h_size)) for v in self.graph.one_hop_neighbors | self.graph.two_hop_neighbors], dim=0)
        f = self.model.forward(adj_matrix, self.graph)
        return f

def calculate_prob(graph, f, target_node):
    nodes_n1_n2 = graph.fn[target_node] | graph.sn[target_node]
    if not nodes_n1_n2:
        return 0.0
    sum_prob = 0.0
    for v_j in nodes_n1_n2:
        if (target_node, v_j) in graph.p or (v_j, target_node) in graph.p:
            sum_prob += f[graph.v.index(v_j)].item()
    probability = sum_prob / len(nodes_n1_n2) if nodes_n1_n2 else 0.0
    return probability

def calculate_expected_prob(cg, P_do, label_probs):
    expected_value = 0.0
    for y, y_prob in label_probs.items():
        inner_sum = 0.0
        for v_i in cg.one_hop_neighbors:
            inner_sum += P_do
        expected_value += y_prob * inner_sum
    return expected_value / len(cg.one_hop_neighbors) if cg.one_hop_neighbors else 0.0

def compute_probability_of_node_label(cg, target_node, role_id):
    unique_labels = np.unique(list(role_id.values()))
    num_nodes = len(cg.v)
    all_combinations = product(unique_labels, repeat=num_nodes)
    label_probabilities = {label: 0 for label in unique_labels}
    for combination in all_combinations:
        temp_role_id = list(combination)
        current_label = temp_role_id[cg.v.index(target_node)]
        label_probabilities[current_label] += 1
    total_combinations = len(unique_labels) ** num_nodes
    for label, count in label_probabilities.items():
        label_probabilities[label] = count / total_combinations
    return label_probabilities

def bpr_loss(pos_pred, neg_pred):
    return -torch.mean(torch.log(torch.sigmoid(pos_pred - neg_pred)))

def train(cg, learning_rate, h_size, h_layers, num_epochs, data, role_id, target_node, multimodal_features, config):
    cg.target_node, cg.one_hop_neighbors, cg.two_hop_neighbors, cg.out_of_neighborhood = cg.categorize_neighbors(target_node=target_node)
    ncm = NCM(cg, target_node, learning_rate, h_size, h_layers, data, multimodal_features, config)
    optimizer = optim.Adam(ncm.model.parameters(), lr=learning_rate)
    new_v = {cg.target_node}.union(cg.one_hop_neighbors)
    loss_history = []
    node_features = torch.stack([
        torch.cat([multimodal_features[v]['text'], multimodal_features[v]['image']], dim=-1) if v in multimodal_features else torch.zeros(h_size)
        for v in cg.v
    ])
    adj = cg.norm_adj
    for i in range(num_epochs):
        f = ncm.ncm_forward(node_features, adj, add_noise=True)
        P_do = calculate_prob(cg, f, cg.target_node)
        label_probs = compute_probability_of_node_label(cg, cg.target_node, role_id)
        expected_p = calculate_expected_prob(cg, P_do, label_probs)
        expected_p_tensor = torch.tensor([expected_p], dtype=torch.float32) if isinstance(expected_p, float) else expected_p
        output = (expected_p_tensor.clone().detach() >= 0.05).float()
        pos_items = list(cg.fn[target_node])
        neg_items = list(cg.on[target_node])
        if pos_items and neg_items:
            pos_pred = f[cg.v.index(pos_items[0])]
            neg_pred = f[cg.v.index(neg_items[0])]
            batch_mf_loss = bpr_loss(pos_pred, neg_pred)
            cl_loss = (ncm.model.InfoNCE(ncm.dropout(f[:cg.n_users]), ncm.dropout(f[:cg.n_users]), 0.2) +
                       ncm.model.InfoNCE(ncm.dropout(f[cg.n_users:]), ncm.dropout(f[cg.n_users:]), 0.2)) / 2
            loss = batch_mf_loss + config['reg_weight'] * cl_loss
        else:
            loss = torch.nn.functional.binary_cross_entropy(f[cg.v.index(target_node)].view(1), torch.tensor([role_id.get(target_node, 0.0)], dtype=torch.float).view(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    return loss_history, loss, ncm.model.state_dict(), expected_p, output, new_v

def print_expected_p_for_each_node(models):
    for node, model_info in models.items():
        expected_p = model_info['expected_p']
        print(f"Node: {node}, Expected Probability: {expected_p}")

def alg_2(Graph, num_epochs, data, role_id, multimodal_features, config, top_k=10):
    if num_epochs is None:
        num_epochs = 100
    models = {}
    for node in Graph.v:
        if node.startswith('user_'):
            Graph.target_node = node
            loss_history, total_loss, model, expected_p, output, new_v = train(
                Graph, config['learning_rate'], config['embedding_size'], config['n_ui_layers'], num_epochs, data, role_id, node, multimodal_features, config
            )
            models[node] = {
                'model': model,
                'expected_p': expected_p,
                'total_loss': total_loss,
                'output': output,
                'new_v': new_v,
                'loss_history': loss_history
            }
    recommendations = {}
    for user in models:
        node_features = torch.stack([
            torch.cat([Graph.multimodal_features[v]['text'], Graph.multimodal_features[v]['image']], dim=-1) if v in Graph.multimodal_features else torch.zeros(config['embedding_size'])
            for v in Graph.v
        ])
        model = CMGNN(config, train_data=Graph, v_feat=multimodal_features.get('v_feat'), t_feat=multimodal_features.get('t_feat'), device=config.get('device', 'cuda'))
        model.load_state_dict(models[user]['model'])
        model.eval()
        with torch.no_grad():
            user_emb, item_emb = model.forward(Graph.norm_adj, Graph)
            u_embeddings = user_emb[Graph.v.index(user)]
            scores = torch.matmul(u_embeddings, item_emb.transpose(0, 1))
        interacted_items = Graph.fn[user]
        candidate_items = [v for v in Graph.v if v not in interacted_items and v.startswith('item_')]
        scores = [(item, scores[Graph.v.index(item)].item()) for item in candidate_items]
        scores.sort(key=lambda x: x[1], reverse=True)
        recommendations[user] = [item for item, _ in scores[:top_k]]
    print_expected_p_for_each_node(models)
    return recommendations, models

def evaluate_recommendations(recommendations, ground_truth, k=10):
    precision, recall, ndcg = [], [], []
    for user in recommendations:
        rec_items = recommendations[user][:k]
        true_items = ground_truth.get(user, [])
        if not true_items:
            continue
        hits = len(set(rec_items) & set(true_items))
        precision.append(hits / k)
        recall.append(hits / len(true_items))
        dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(rec_items) if item in true_items)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(true_items))))
        ndcg.append(dcg / idcg if idcg > 0 else 0)
    return {
        'Precision@K': np.mean(precision) if precision else 0.0,
        'Recall@K': np.mean(recall) if recall else 0.0,
        'NDCG@K': np.mean(ndcg) if ndcg else 0.0
    }