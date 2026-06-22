import torch
import torch.nn as nn
import dgl
import numpy as np
import random
import scipy.sparse as sp
from collections import defaultdict

from .get_relation_map import get_relation_map


class GuardianInjector(nn.Module):
    def __init__(
        self,
        dataname,
        noise_std=0.01,
        model=None,
        h_dict=None,
        meta_paths=None,
        feature_dim=None,
        proxy_model=None,
    ):
        super().__init__()

        self.noise_std = noise_std
        self.model = model
        self.h_dict = h_dict
        self.meta_paths = meta_paths
        self.feature_dim = feature_dim
        self.proxy_model = proxy_model

        self.last_guardian_nodes = []
        self.injected_path_names = {}
        self.guardian_feat_dict = defaultdict(list)
        self.last_hete_adjs = {}

        self.relation_map = get_relation_map(dataname)

    def inject(
        self,
        hg,
        sensitive_nodes,
        feat_key="x",
        target_type=None,
        sensitive_node_paths=None,
        top_k_ratio=None,
        drop_ratio=None,
    ):
        device = hg.device

        if target_type is None:
            raise ValueError("target_type must be provided.")

        if torch.is_tensor(sensitive_nodes):
            sensitive_nodes = sensitive_nodes.detach().cpu().tolist()
        else:
            sensitive_nodes = list(sensitive_nodes)

        sensitive_nodes = list(map(int, sensitive_nodes))

        for ntype in hg.ntypes:
            if dgl.NID not in hg.nodes[ntype].data:
                hg.nodes[ntype].data[dgl.NID] = torch.arange(
                    hg.num_nodes(ntype),
                    device=device,
                )

        new_feats = {
            ntype: hg.nodes[ntype].data[feat_key].clone().to(device)
            for ntype in hg.ntypes
            if feat_key in hg.nodes[ntype].data
        }

        if target_type not in new_feats:
            raise KeyError(
                f"Feature key '{feat_key}' is not found for target node type '{target_type}'."
            )

        edge_buffer = defaultdict(lambda: {"src": [], "dst": []})
        guardian_edges = set()
        guardian_ids = []

        for sensitive_id in sensitive_nodes:
            if sensitive_node_paths is None:
                continue

            if sensitive_id not in sensitive_node_paths:
                continue

            best_path, _ = sensitive_node_paths[sensitive_id]

            if not best_path:
                continue

            first_rel = best_path[0]

            if first_rel not in self.relation_map:
                continue

            src_type, rel, dst_type = self.relation_map[first_rel]

            if src_type == target_type and dst_type != target_type:
                proxy_type = dst_type
                neighbors = hg.successors(
                    sensitive_id,
                    etype=(src_type, rel, dst_type),
                ).detach().cpu().tolist()

            elif dst_type == target_type and src_type != target_type:
                proxy_type = src_type
                neighbors = hg.predecessors(
                    sensitive_id,
                    etype=(src_type, rel, dst_type),
                ).detach().cpu().tolist()

            else:
                continue

            if len(neighbors) == 0:
                continue

            proxy_local_id = random.choice(neighbors)

            origin_feat = new_feats[target_type][sensitive_id]
            noise = torch.randn_like(origin_feat) * self.noise_std
            guardian_tensor = origin_feat + noise

            guardian_param = nn.Parameter(
                guardian_tensor.clone(),
                requires_grad=True,
            )

            guardian_id = new_feats[target_type].shape[0]

            self.register_parameter(
                f"guardian_feat_{target_type}_{guardian_id}",
                guardian_param,
            )

            new_feats[target_type] = torch.cat(
                [
                    new_feats[target_type],
                    guardian_tensor.unsqueeze(0).detach(),
                ],
                dim=0,
            )

            self.guardian_feat_dict[target_type].append(guardian_param)
            guardian_ids.append(guardian_id)

            etype = (target_type, rel, proxy_type)

            edge_buffer[etype]["src"].append(guardian_id)
            edge_buffer[etype]["dst"].append(proxy_local_id)

            guardian_edges.add(
                (etype, guardian_id, proxy_local_id)
            )

            self.injected_path_names[str(sensitive_id)] = "".join(best_path)

        data_dict = {
            etype: (
                hg.edges(etype=etype, form="uv")[0].to(device),
                hg.edges(etype=etype, form="uv")[1].to(device),
            )
            for etype in hg.canonical_etypes
        }

        for etype, edict in edge_buffer.items():
            src = torch.tensor(
                edict["src"],
                dtype=torch.long,
                device=device,
            )
            dst = torch.tensor(
                edict["dst"],
                dtype=torch.long,
                device=device,
            )

            if etype in data_dict:
                data_dict[etype] = (
                    torch.cat([data_dict[etype][0], src]),
                    torch.cat([data_dict[etype][1], dst]),
                )
            else:
                data_dict[etype] = (src, dst)

        num_nodes_dict = {}

        for ntype in hg.ntypes:
            feat_num = (
                new_feats[ntype].shape[0]
                if ntype in new_feats
                else hg.num_nodes(ntype)
            )

            max_edge_id = -1

            for (src_type, _, dst_type), edict in edge_buffer.items():
                if src_type == ntype and len(edict["src"]) > 0:
                    max_edge_id = max(max_edge_id, max(edict["src"]))

                if dst_type == ntype and len(edict["dst"]) > 0:
                    max_edge_id = max(max_edge_id, max(edict["dst"]))

            num_nodes_dict[ntype] = max(
                feat_num,
                hg.num_nodes(ntype),
                max_edge_id + 1,
            )

        new_g = dgl.heterograph(
            data_dict,
            num_nodes_dict=num_nodes_dict,
        ).to(device)

        for ntype in new_feats:
            new_g.nodes[ntype].data[feat_key] = new_feats[ntype]
            new_g.nodes[ntype].data[dgl.NID] = torch.arange(
                new_feats[ntype].shape[0],
                device=device,
            )

        for etype in new_g.canonical_etypes:
            num_edges = new_g.num_edges(etype)
            edge_mask = torch.zeros(
                num_edges,
                dtype=torch.bool,
                device=device,
            )

            src, dst = new_g.edges(etype=etype)

            for i in range(num_edges):
                edge_key = (
                    etype,
                    src[i].item(),
                    dst[i].item(),
                )
                if edge_key in guardian_edges:
                    edge_mask[i] = True

            new_g.edges[etype].data["guardian_edge_mask"] = edge_mask

        self.last_guardian_nodes = guardian_ids

        hete_adjs = {}

        for etype in new_g.canonical_etypes:
            mask = ~new_g.edges[etype].data["guardian_edge_mask"]
            src, dst = new_g.edges(etype=etype, form="uv")

            src_np = src[mask].detach().cpu().numpy()
            dst_np = dst[mask].detach().cpu().numpy()

            adj = sp.csr_matrix(
                (
                    np.ones_like(src_np),
                    (src_np, dst_np),
                ),
                shape=(
                    new_g.num_nodes(etype[0]),
                    new_g.num_nodes(etype[2]),
                ),
            )

            short_etype = etype[0][0] + etype[2][0]
            hete_adjs[short_etype] = adj
            hete_adjs[short_etype[::-1]] = adj.transpose()

        self.last_hete_adjs = hete_adjs

        num_total = new_g.num_nodes(target_type)

        guardian_mask = torch.zeros(
            num_total,
            dtype=torch.bool,
            device=device,
        )

        if len(guardian_ids) > 0:
            guardian_mask[
                torch.tensor(
                    guardian_ids,
                    dtype=torch.long,
                    device=device,
                )
            ] = True

        new_g.nodes[target_type].data["guardian_mask"] = guardian_mask

        if "y" in hg.nodes[target_type].data:
            old_labels = hg.nodes[target_type].data["y"].to(device)

            padded_labels = torch.cat(
                [
                    old_labels,
                    torch.full(
                        (len(guardian_ids),),
                        -1,
                        dtype=old_labels.dtype,
                        device=device,
                    ),
                ],
                dim=0,
            )

            new_g.nodes[target_type].data["y"] = padded_labels
            self.num_orig_nodes = old_labels.shape[0]

        else:
            new_g.nodes[target_type].data["y"] = -1 * torch.ones(
                num_total,
                dtype=torch.long,
                device=device,
            )
            self.num_orig_nodes = 0

        return (
            new_g,
            new_feats,
            self.injected_path_names,
            edge_buffer,
        )