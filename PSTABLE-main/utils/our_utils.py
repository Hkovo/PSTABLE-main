import numpy as np
import scipy.sparse as sp
import dgl
import torch
from torch_geometric.utils import coalesce, scatter
import torch.nn.functional as F


def eidx2spmatrix(edge_index, edge_weight=None, n_nodes=None, message_passing=False):
    if edge_weight is None:
        edge_weight = np.ones(edge_index.size(1))
    else:
        edge_weight = edge_weight.detach().cpu().numpy()
    if message_passing:
        edge_index = edge_index[[1, 0]].detach().cpu().numpy()
    else:
        edge_index = edge_index.detach().cpu().numpy()
    if n_nodes is None:
        n_nodes = edge_index.max().item() + 1
    return sp.csr_matrix((edge_weight, edge_index), shape=(n_nodes, n_nodes))


def get_homo_adj_list(given_hete_adjs, metapath_info):
    # 用来存储归一化后的异质图邻接矩阵
    hete_adj_dict_tmp = {}

    # 对每个异质图邻接矩阵进行归一化处理，计算度数并进行归一化是为了平衡不同节点之间的信息传递，使得模型更容易学习到全局信息
    for key, adj in given_hete_adjs.items():
        # 计算度数
        deg = adj.sum(axis=1).A1  # 使用 .A1 转为 1D 数组
        # 归一化邻接矩阵，确保度大于0时进行归一化
        deg = np.where(deg > 0, deg, 1)  # 保证度不为0
        # hete_adj_dict_tmp[key] = adj / deg[:, None]  # 按行归一化
        hete_adj_dict_tmp[key] = sp.diags(1. / deg).dot(adj)

    # 保存生成的同质图邻接矩阵列表
    homo_adj_list = []
    # 根据元路径逐步生成同质图的邻接矩阵
    for i in range(len(metapath_info)):
        # 获取第一个边类型的邻接矩阵
        adj = hete_adj_dict_tmp[metapath_info[i][0]]

        # 按照元路径计算同质图的邻接矩阵
        for etype in metapath_info[i][1:]:
            adj = adj.dot(hete_adj_dict_tmp[etype])  # 矩阵乘法
            # print(f"After multiplying {etype}, shape of adj: {adj.shape}")

        # 将生成的矩阵转换为稀疏矩阵
        homo_adj_list.append(sp.csr_matrix(adj))  # 转为稀疏矩阵
        # print(adj)
    return homo_adj_list  # 返回同质图邻接矩阵列表


def get_transition(homo_adj_list, x):
    device = x.device
    edata_list = []
    # x_norm = F.normalize(x).cpu()
    # NOTE: inner prod 而不是 sim
    x_norm = x.cpu()
    sim_mat = torch.mm(x_norm, x_norm.t())
    for i, adj in enumerate(homo_adj_list):
        adj = adj.tocoo()
        edge_index = torch.LongTensor(np.stack((adj.row, adj.col), 0))
        edge_weight = torch.FloatTensor(adj.data)
        # 计算边相似度
        edge_sim = sim_mat[edge_index[0], edge_index[1]]
        edge_weight = edge_weight * edge_sim
        row, col = edge_index[0], edge_index[1]
        # 入度
        deg = scatter(edge_weight, row, dim=0, dim_size=x.size(0), reduce='sum')
        # NOTE: -0.5对称归一化，原本实现是-1对称归一化
        # deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt = deg.pow(-1.0)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 1.0
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        edata_list.append((edge_index.to(device), edge_weight.to(device), x.shape[0]))
    return edata_list


def get_homo_trans_data(x, hete_adjs, metapath_info):
    """
    Returns
        trans_edata_list: 每个 edata 是一个三元组 (edge_index, edge_weight, n_nodes)
    """
    homo_adjs = get_homo_adj_list(hete_adjs, metapath_info)
    return get_transition(homo_adjs, x)


def get_purified_graphs(trans_edata_list, thrds, dtype='dgl'):
    device = trans_edata_list[0][0].device
    homo_gs = []
    for i, (eidx, ewgt, n_nodes) in enumerate(trans_edata_list):
        mask = ewgt >= thrds[i]
        # NOTE: GAT不需要边权，测试调整边权
        purified_adj = eidx2spmatrix(eidx[:, mask], ewgt[mask], n_nodes).tocoo()
        # purified_adj = eidx2spmatrix(eidx[:, mask], None, n_nodes).tocoo()
        if dtype == 'dgl':
            homo_gs.append(dgl.from_scipy(purified_adj, eweight_name='weight').to(device))
        else:
            homo_gs.append(purified_adj)
    return homo_gs


def get_semantic_weighted_matrix(trans_edata_list, sem_attention, thrd):
    assert len(trans_edata_list) == len(sem_attention)
    trans_eidx_list = [edata[0] for edata in trans_edata_list]
    trans_ewgt_list = [edata[1] for edata in trans_edata_list]
    weighted_ewgt_list = [trans_ewgt_list[i] * sem_attention[i] for i in range(len(sem_attention))]
    new_eidx, new_ewgt = coalesce(torch.cat(trans_eidx_list, dim=1), torch.cat(weighted_ewgt_list, dim=0), reduce='sum')

    mask = new_ewgt >= thrd
    new_eidx = new_eidx[:, mask]
    new_ewgt = new_ewgt[mask]

    N = trans_edata_list[0][2]
    new_matrix = eidx2spmatrix(new_eidx, new_ewgt, n_nodes=N, message_passing=True)
    return new_matrix


def perturb_sparse_matrix(matrix, noise_rate):
    """
    随机扰动稀疏矩阵的拓扑结构，通过打乱部分边的目标节点。
    
    Args:
        matrix (sp.csr_matrix): 输入的稀疏矩阵（CSR格式）。
        noise_rate (float): 扰动率（需要扰动的边比例）。
        
    Returns:
        sp.csr_matrix: 扰动后的稀疏矩阵。
    """
    # 确保输入是稀疏矩阵
    if not sp.issparse(matrix):
        raise ValueError("Input matrix must be a sparse matrix.")
    
    # 获取稀疏矩阵的非零元素索引和数据
    row, col = matrix.nonzero()  # 获取行索引和列索引
    data = matrix.data  # 边权重
    
    # 计算需要扰动的边数量
    num_edges = len(data)
    num_perturb = int(noise_rate * num_edges)
    
    if num_perturb == 0:
        # 如果扰动边数为0，直接返回原始矩阵
        return matrix
    
    # 随机选择要扰动的边的索引
    perturb_indices = np.random.choice(num_edges, num_perturb, replace=False)
    
    # 获取这些边的原始行索引和列索引
    perturb_rows = row[perturb_indices]
    perturb_cols = col[perturb_indices]
    
    # 对列索引（目标节点）进行随机打乱（洗牌）
    shuffled_cols = np.random.permutation(perturb_cols)
    
    # 创建新的行索引和列索引
    new_row = np.copy(row)
    new_col = np.copy(col)
    
    # 替换扰动的列索引
    new_col[perturb_indices] = shuffled_cols
    
    # 生成新的稀疏矩阵
    perturbed_matrix = sp.csr_matrix((data, (new_row, new_col)), shape=matrix.shape)
    
    return perturbed_matrix
