from sklearn.metrics import f1_score
import datetime
import dgl
import numpy as np
import os
import random
import torch
import scipy.sparse as sp
import time

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return accuracy, micro_f1, macro_f1

def score_detail(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    acc_detail = np.array(prediction == labels, dtype='int')
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return accuracy, micro_f1, macro_f1, acc_detail

def evaluate(model, adj_matrix, features, labels, mask, loss_func, detail=False):
    model.eval()
    with torch.no_grad():
        if model.__class__.__name__ == "GC_HAN_AC":
            type_mask = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
            inputs1 = (None, [features], type_mask)
            logits, _, _ = model(inputs1)
        else:
            logits = model(features, adj_matrix)
    loss = loss_func(logits[mask], labels[mask])
    if detail:
        accuracy, micro_f1, macro_f1, acc_detail = score_detail(logits[mask], labels[mask])
        return acc_detail, accuracy, micro_f1, macro_f1
    else:
        accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])
        return loss, accuracy, micro_f1, macro_f1

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_date_postfix():
    dt = datetime.datetime.now()
    return '{}_{:02d}-{:02d}-{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second)

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size, dtype=torch.bool)
    mask[indices] = 1
    return mask

def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        save_dir = os.path.dirname(self.filename)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.filename))

def to_tensor(adj, features, labels=None, device='cpu'):
    if sp.issparse(adj):
        adj = sparse_mx_to_sparse_tensor(adj)
    else:
        adj = torch.FloatTensor(adj)
    if sp.issparse(features):
        features = sparse_mx_to_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features))
    if labels is None:
        return adj.to(device), features.to(device)
    else:
        labels = torch.LongTensor(labels)
        return adj.to(device), features.to(device), labels.to(device)

def sparse_mx_to_sparse_tensor(sparse_mx):
    sparse_mx_coo = sparse_mx.tocoo().astype(np.float32)
    sparse_row = torch.LongTensor(sparse_mx_coo.row).unsqueeze(1)
    sparse_col = torch.LongTensor(sparse_mx_coo.col).unsqueeze(1)
    sparse_indices = torch.cat((sparse_row, sparse_col), 1)
    sparse_data = torch.FloatTensor(sparse_mx.data)
    return torch.sparse_coo_tensor(sparse_indices.t(), sparse_data, size=sparse_mx.shape, dtype=torch.float)

def log_result(filepath, dataname, atk_name, atk_rate, model, vals):
    log_dir = os.path.dirname(filepath)
    if log_dir != "":
        os.makedirs(log_dir, exist_ok=True)
    with open(filepath, mode='a+', encoding='utf-8') as f:
        vals = ', '.join([f"{v:.5f}" for v in vals])
        f.write(f"{dataname} {model:<20} {atk_name} {atk_rate}: {vals}\n")
