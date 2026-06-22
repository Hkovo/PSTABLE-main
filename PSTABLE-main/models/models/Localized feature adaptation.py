import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm


class GuardianFineTuner:
    def __init__(
        self,
        model,
        data,
        guardian_feat_params,
        guardian_nodes,
        ground_score,
        sel_train_nodes,   # 允许保留，但不会作为唯一依据
        val_nodes,
        inf_score,
        easy_score,
        sensitive_nodes,
        lambda_=0.1,
        lr=0.01,
        epochs=50,
        device="cuda",
        target_type=None,
        alpha=1e-3,
        s_test=None,
        y_pseudo_test=None,
        use_hinge=False,
        hinge_margin=1.0,
        use_train_mask_from_graph=True,  # ✅ 新增：默认用注入后图的 train_mask
    ):
        self.model = model.to(device)
        self.data = data.to(device)
        self.target_type = target_type
        self.device = device

        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.lambda_ = lambda_
        self.use_hinge = use_hinge
        self.hinge_margin = hinge_margin
        self.use_train_mask_from_graph = use_train_mask_from_graph

        guardian_info = sorted(zip(guardian_nodes, guardian_feat_params), key=lambda x: x[0])
        self.guardian_nodes = torch.as_tensor(
            [int(nid) for nid, _ in guardian_info], dtype=torch.long, device=device
        )

        self.guardian_feats = nn.ParameterList(
            [nn.Parameter(f.clone().detach().to(device).requires_grad_()) for _, f in guardian_info]
        )
        self.original_feats = [p.data.clone().detach() for p in self.guardian_feats]
        self.optimizer = Adam(self.guardian_feats.parameters(), lr=self.lr)

        for p in self.model.parameters():
            p.requires_grad = False

        self.sensitive_nodes = torch.as_tensor(sensitive_nodes, dtype=torch.long, device=device)
        self.sel_train_nodes = torch.as_tensor(sel_train_nodes, dtype=torch.long, device=device) if sel_train_nodes is not None else None
        self.val_nodes = torch.as_tensor(val_nodes, dtype=torch.long, device=device) if val_nodes is not None else None

        self.ground_score = ground_score.to(device) if ground_score is not None else None
        self.inf_score = inf_score.to(device) if inf_score is not None else None

        # ✅ 强烈建议：Guardian 微调阶段默认不要用 easy_score
        self.easy_score = easy_score.to(device) if easy_score is not None else None

        labels_all = self.data.nodes[self.target_type].data["y"].to(device).clone()
        if (s_test is not None) and (y_pseudo_test is not None):
            s_test = torch.as_tensor(s_test, dtype=torch.long, device=device)
            y_pseudo_test = torch.as_tensor(y_pseudo_test, dtype=torch.long, device=device)
            labels_all[s_test] = y_pseudo_test
            self.s_test = s_test
        else:
            self.s_test = None
        self.ground_labels = labels_all

    def _hinge_margin_loss(self, logits, y, margin_m=1.0, w=None):
        valid = y >= 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        logits = logits[valid]
        y = y[valid]
        if w is not None:
            w = w[valid]

        idx = torch.arange(logits.size(0), device=logits.device)
        pos = logits[idx, y]
        tmp = logits.clone()
        tmp[idx, y] = -1e9
        neg = tmp.max(dim=1).values
        loss = (margin_m - (pos - neg)).clamp_min(0.0)
        if w is not None:
            loss = loss * w
        return loss.mean()

    def _ce_loss(self, logits, y, w=None):
        valid = y >= 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        logits = logits[valid]
        y = y[valid]
        if w is not None:
            w = w[valid]

        ce = F.cross_entropy(logits, y, reduction="none")
        if w is not None:
            ce = ce * w
        return ce.mean()

    def _get_train_mask(self):
        # ✅ 优先用注入后图里的 train_mask，避免 sel_train_nodes(旧图)错位
        if self.use_train_mask_from_graph and "train_mask" in self.data.nodes[self.target_type].data:
            return self.data.nodes[self.target_type].data["train_mask"].to(self.device).bool()
        # fallback：用 sel_train_nodes 构 mask
        train_mask = torch.zeros_like(self.ground_labels, dtype=torch.bool, device=self.device)
        if self.sel_train_nodes is not None and self.sel_train_nodes.numel() > 0:
            train_mask[self.sel_train_nodes] = True
        return train_mask

    def proactive_objective(self, logits):
        labels_all = self.ground_labels
        train_mask = self._get_train_mask()

        if self.sensitive_nodes.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        protect_nodes = self.sensitive_nodes[train_mask[self.sensitive_nodes]]
        if protect_nodes.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        y = labels_all[protect_nodes]
        valid = y >= 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        # ✅ 权重：如果全 0，就直接禁用，避免 loss 被乘没
        weights = None
        if self.easy_score is not None:
            w = self.easy_score[protect_nodes][valid]
            if w.mean().abs() >= 1e-12:
                weights = w / (w.mean().clamp_min(1e-12))
            else:
                weights = None  # 关键：全 0 -> 不加权

        if self.use_hinge:
            ce_term = self._hinge_margin_loss(logits[protect_nodes][valid], y[valid],
                                              margin_m=self.hinge_margin, w=weights)
        else:
            ce_term = self._ce_loss(logits[protect_nodes][valid], y[valid], w=weights)

        if len(self.guardian_feats) == 0:
            reg_term = torch.tensor(0.0, device=self.device)
        else:
            reg_term = torch.stack([(p - orig).norm(p=2) for p, orig in zip(self.guardian_feats, self.original_feats)]).sum()

        return ce_term + self.alpha * reg_term

    def fine_tune(self):
        best_loss = float("inf")
        best_feats = [p.data.clone().detach() for p in self.guardian_feats]
        self.model.eval()

        train_mask = self._get_train_mask()
        protect_count = int(self.sensitive_nodes[train_mask[self.sensitive_nodes]].numel())

        for epoch in tqdm(range(self.epochs), desc="Guardian Fine-tuning"):
            self.optimizer.zero_grad()

            base_features = self.data.nodes[self.target_type].data["x"].to(self.device).clone()
            guardian_tensor = torch.stack(list(self.guardian_feats), dim=0)
            base_features[self.guardian_nodes] = guardian_tensor
            self.data.nodes[self.target_type].data["x"] = base_features

            logits = self.model({self.target_type: base_features}, self.data)
            loss = self.proactive_objective(logits)

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f"[Epoch {epoch}] Loss = {loss.item():.6f} | protect(train∩sensitive)={protect_count}")
                if protect_count > 0 and abs(loss.item()) < 1e-12:
                    print("⚠️  Loss≈0 while protect_nodes>0: check weights/easy_score or labels validity.")

            loss.backward()
            self.optimizer.step()

            self.data.nodes[self.target_type].data["x"] = self.data.nodes[self.target_type].data["x"].detach()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_feats = [p.data.clone().detach() for p in self.guardian_feats]

        for p, bf in zip(self.guardian_feats, best_feats):
            p.data.copy_(bf)

        print("✅ Final best loss:", best_loss)
        return torch.stack(best_feats).detach()
