import os
import os.path as osp
from pathlib import Path

import dgl
import numpy as np
import scipy.sparse as sp
import torch
from scipy import io as sio
from torch_geometric.datasets import DBLP

from core.config import predict_ntype_dict
from utils.utils import set_random_seed, get_binary_mask


ROOT = Path(__file__).resolve().parents[1]


def load_acm_raw(remove_self_loop=False):
    assert not remove_self_loop

    set_random_seed(1)

    data_path = ROOT / "dataset" / "ACM" / "ACM.mat"

    if not data_path.exists():
        raise FileNotFoundError(f"ACM data file not found: {data_path}")

    data = sio.loadmat(data_path)

    p_vs_f = data["PvsL"]
    p_vs_a = data["PvsA"]
    p_vs_t = data["PvsT"]
    p_vs_c = data["PvsC"]

    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]

    p_vs_f = p_vs_f[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]

    n_node_dict = {
        "paper": p_vs_f.shape[0],
        "field": p_vs_f.shape[1],
        "author": p_vs_a.shape[1],
    }

    hete_adjs = {
        "pa": p_vs_a,
        "ap": p_vs_a.T,
        "pf": p_vs_f,
        "fp": p_vs_f.T,
    }

    hg = dgl.heterograph(
        {
            ("paper", "pa", "author"): p_vs_a.nonzero(),
            ("author", "ap", "paper"): p_vs_a.T.nonzero(),
            ("paper", "pf", "field"): p_vs_f.nonzero(),
            ("field", "fp", "paper"): p_vs_f.T.nonzero(),
        },
        num_nodes_dict=n_node_dict,
    )

    features = torch.FloatTensor(p_vs_t.toarray())
    hg.ndata["x"] = {"paper": features}

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)

    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id

    labels = torch.LongTensor(labels)
    num_classes = 3

    return hg, hete_adjs, features, labels, num_classes


def load_dblp_raw(remove_self_loop=False):
    assert not remove_self_loop

    set_random_seed(1)

    data_root = ROOT / "dataset" / "DBLP"
    dataset = DBLP(str(data_root))
    data = dataset[0]

    num_classes = int(data["author"].y.max().item()) + 1
    features = data["author"].x
    labels = data["author"].y

    author_paper_edge_index = data["author", "to", "paper"].edge_index
    rows = author_paper_edge_index[0].cpu().numpy()
    cols = author_paper_edge_index[1].cpu().numpy()
    ap = sp.csr_matrix((np.ones(len(rows)), (rows, cols)))

    paper_term_edge_index = data["paper", "to", "term"].edge_index
    rows = paper_term_edge_index[0].cpu().numpy()
    cols = paper_term_edge_index[1].cpu().numpy()
    pt = sp.csr_matrix((np.ones(len(rows)), (rows, cols)))

    paper_conference_edge_index = data["paper", "to", "conference"].edge_index
    rows = paper_conference_edge_index[0].cpu().numpy()
    cols = paper_conference_edge_index[1].cpu().numpy()
    pc = sp.csr_matrix((np.ones(len(rows)), (rows, cols)))

    hete_adjs = {
        "ap": ap,
        "pa": ap.T,
        "pc": pc,
        "cp": pc.T,
        "pt": pt,
        "tp": pt.T,
    }

    hg = dgl.heterograph(
        {
            ("author", "ap", "paper"): ap.nonzero(),
            ("paper", "pa", "author"): ap.T.nonzero(),
            ("paper", "pc", "conf"): pc.nonzero(),
            ("conf", "cp", "paper"): pc.T.nonzero(),
            ("paper", "pt", "term"): pt.nonzero(),
            ("term", "tp", "paper"): pt.T.nonzero(),
        }
    )

    features = torch.FloatTensor(features)
    hg.ndata["x"] = {"author": features}

    return hg, hete_adjs, features, labels, num_classes


def load_yelp_raw(remove_self_loop=False):
    assert not remove_self_loop

    set_random_seed(1)

    data_path = ROOT / "dataset" / "YELP" / "YELP.pt"

    if not data_path.exists():
        raise FileNotFoundError(f"YELP data file not found: {data_path}")

    data = torch.load(data_path, map_location="cpu")

    pred_ntype = "b"
    features = data[pred_ntype].x
    labels = data[pred_ntype].y
    num_classes = int(labels.max().item()) + 1

    hete_adjs = {}
    graph_data = {}
    num_nodes_dict = {
        ntype: data[ntype].x.shape[0]
        for ntype in data.node_types
    }

    for etype in data.edge_types:
        src_type, rel_type, dst_type = etype
        edge_index = data[etype].edge_index

        rows = edge_index[0].cpu().numpy()
        cols = edge_index[1].cpu().numpy()

        adj = sp.csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(num_nodes_dict[src_type], num_nodes_dict[dst_type]),
        )

        short_etype = src_type[0] + dst_type[0]
        hete_adjs[short_etype] = adj
        hete_adjs[short_etype[::-1]] = adj.T

        graph_data[(src_type, rel_type, dst_type)] = (
            torch.tensor(rows, dtype=torch.int64),
            torch.tensor(cols, dtype=torch.int64),
        )

    hg = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
    hg.ndata["x"] = {pred_ntype: features}

    return hg, hete_adjs, features, labels, num_classes


def random_split_train_val_test(num_nodes, train_ratio, val_ratio, test_ratio):
    train_size = int(num_nodes * train_ratio / (train_ratio + val_ratio + test_ratio))
    val_size = int(num_nodes * val_ratio / (train_ratio + val_ratio + test_ratio))
    test_size = num_nodes - train_size - val_size

    rs = np.random.RandomState(1)
    indices = rs.permutation(num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size: train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    return train_mask, val_mask, test_mask


def load_clean_data(dataname, skip_enhanced=True):
    if dataname == "acm":
        hg, hete_adjs, features, labels, num_classes = load_acm_raw(False)

        num_nodes = features.shape[0]
        float_mask = np.zeros(num_nodes)

        for class_id in range(num_classes):
            class_mask = (labels == class_id).cpu().numpy()
            float_mask[class_mask] = np.random.permutation(
                np.linspace(0, 1, class_mask.sum())
            )

        train_idx = np.where(float_mask <= 0.2)[0]
        val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
        test_idx = np.where(float_mask > 0.3)[0]

        train_mask = get_binary_mask(num_nodes, train_idx)
        val_mask = get_binary_mask(num_nodes, val_idx)
        test_mask = get_binary_mask(num_nodes, test_idx)

    elif dataname == "dblp":
        hg, hete_adjs, features, labels, num_classes = load_dblp_raw(False)
        train_mask, val_mask, test_mask = random_split_train_val_test(
            features.shape[0], 1, 1, 8
        )

    elif dataname == "yelp":
        hg, hete_adjs, features, labels, num_classes = load_yelp_raw(False)
        train_mask, val_mask, test_mask = random_split_train_val_test(
            features.shape[0], 1, 1, 8
        )

    else:
        raise ValueError(f"Unknown dataset: {dataname}")

    return hg, hete_adjs, features, labels, num_classes, train_mask, val_mask, test_mask


def load_perturbed_data(dataname, atk_name, atk_rate):
    pert_data = None

    hg, hete_adjs, features, labels, num_classes, train_mask, val_mask, test_mask = load_clean_data(
        dataname,
        skip_enhanced=True,
    )

    if atk_rate == 0:
        return hg, hete_adjs, features, labels, num_classes, train_mask, val_mask, test_mask, pert_data

    supported_attacks = ["PRBCD", "HetePRBCD", "FGSM"]

    if atk_name not in supported_attacks:
        raise ValueError(
            f"Unsupported attack type: {atk_name}. "
            f"Supported attacks: {supported_attacks}"
        )

    data_local_dir = ROOT / "data" / f"{dataname}_{atk_name}_{atk_rate}.pt"

    if not data_local_dir.exists():
        raise FileNotFoundError(
            f"Perturbed graph file not found: {data_local_dir}\n"
            f"Please put the pre-generated perturbed graph file under the data/ directory."
        )

    pert_data = torch.load(data_local_dir, map_location="cpu")

    pred_ntype = predict_ntype_dict[dataname]

    features = pert_data[pred_ntype].x
    labels = pert_data[pred_ntype].y
    train_mask = pert_data[pred_ntype].train_mask
    val_mask = pert_data[pred_ntype].val_mask
    test_mask = pert_data[pred_ntype].test_mask

    num_node_dict = {
        ntype: pert_data[ntype].x.shape[0]
        for ntype in pert_data.node_types
    }

    hete_adjs = {}
    graph_data = {}

    for etype in pert_data.edge_types:
        src_type, rel_type, dst_type = etype
        eidx = pert_data[etype]["edge_index"].cpu().numpy()

        if hasattr(pert_data[etype], "edge_weight"):
            ewgt = pert_data[etype].edge_weight.cpu().numpy()
        else:
            ewgt = torch.ones(eidx.shape[1]).numpy()

        adj = sp.csr_matrix(
            (ewgt, (eidx[0], eidx[1])),
            shape=(num_node_dict[src_type], num_node_dict[dst_type]),
        )

        short_etype = src_type[0] + dst_type[0]
        hete_adjs[short_etype] = adj
        hete_adjs[short_etype[::-1]] = adj.T

        graph_data[(src_type, rel_type, dst_type)] = (
            torch.tensor(eidx[0], dtype=torch.int64),
            torch.tensor(eidx[1], dtype=torch.int64),
        )

    hg = dgl.heterograph(
        graph_data,
        num_nodes_dict=num_node_dict,
    )

    return hg, hete_adjs, features, labels, num_classes, train_mask, val_mask, test_mask, pert_data