import numpy as np
import scipy.sparse as sp
import torch
import dgl
from torch_geometric.data import HeteroData


def dglhg2pyghdata(hg: dgl.DGLHeteroGraph, pred_x_dict={}, fill_type='id'):
    data = HeteroData()
    for etype in hg.canonical_etypes:
        data[etype].edge_index = torch.stack(hg[etype].all_edges(), dim=0)
    assert len(pred_x_dict) == 1
    pred_ntype = list(pred_x_dict.keys())[0]
    D = pred_x_dict[pred_ntype].shape[1]
    x_dict = hg.ndata['x'] | pred_x_dict
    assert hg.num_nodes(pred_ntype) == x_dict[pred_ntype].shape[0]
    device = x_dict[pred_ntype].device
    for ntype in hg.ntypes:
        if ntype in x_dict:
            data[ntype].x = x_dict[ntype]
        else:
            N = hg.num_nodes(ntype)
            if fill_type == 'id':
                rows = torch.arange(N)
                filled_vec = torch.sparse_coo_tensor(torch.stack((rows, rows), 0), torch.ones(N), (N, N)).to(device)
            elif fill_type == 'zero':
                filled_vec = torch.zeros(N, D).to(device)
            else:
                continue
            data[ntype].x = filled_vec
    return data


def to_full_heterograph(data: HeteroData):
    node_idx_bias = data.node_offsets
    sorted_ntypes = data.node_types
    N = data.num_nodes

    x_dict = {}
    for ntype in sorted_ntypes:
        x_dict[ntype] = data[ntype].x

    eidx_dict = {}
    for etype in data.edge_types:
        edge_index = data[etype].edge_index
        edge_index[0] += node_idx_bias[etype[0]]
        edge_index[1] += node_idx_bias[etype[-1]]
        eidx_dict[etype] = edge_index
    adj_dict = {k: sp.csr_matrix((np.ones(v.size(1)), v.cpu().numpy()), shape=(N, N)) for k, v in eidx_dict.items()}

    start_slices = {ntype: node_idx_bias[ntype] for ntype in sorted_ntypes}
    end_slices = {ntype: node_idx_bias[ntype] + data[ntype].x.shape[0] for ntype in sorted_ntypes}

    return {
        'x_dict': x_dict,
        'adj_dict': adj_dict,
        'start': start_slices,
        'end': end_slices
    }

def pyghdata2dglhg(data: HeteroData):
    edge_index_dict = {}
    for (src, rel, dst), edge_index in data.edge_index_dict.items():
        src_nodes = edge_index[0].cpu().numpy()
        dst_nodes = edge_index[1].cpu().numpy()
        edge_index_dict[(src, rel, dst)] = (src_nodes, dst_nodes)

    num_nodes_dict = {
        ntype: data[ntype].x.size(0) for ntype in data.node_types
    }

    g = dgl.heterograph(edge_index_dict, num_nodes_dict=num_nodes_dict)
    device = data[data.node_types[0]].x.device
    g = g.to(device)

    for ntype in data.node_types:
        if hasattr(data[ntype], 'x'):
            g.nodes[ntype].data['x'] = data[ntype].x
        if hasattr(data[ntype], 'y'):
            g.nodes[ntype].data['y'] = data[ntype].y
        for mask_name in ['train_mask', 'val_mask', 'test_mask']:
            if hasattr(data[ntype], mask_name):
                g.nodes[ntype].data[mask_name] = getattr(data[ntype], mask_name)

    return g
