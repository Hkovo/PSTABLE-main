import os
import gc
import random
import warnings

import dgl
import numpy as np
import torch
from tqdm import trange

from core.dataloader import load_perturbed_data
from core.config import get_config
from models import PSTABLE
from utils.utils import *

warnings.filterwarnings("ignore", category=FutureWarning)


def build_model(args, dataname, features, pred_ntype, num_classes, device):
    model = PSTABLE(
        meta_paths=args.meta_paths_dict[dataname],
        in_size=features.shape[1],
        hidden_size=args.hidden_units,
        out_size=num_classes,
        num_heads=args.num_heads,
        dropout=args.dropout,
        target_type=pred_ntype,
    )
    return model.to(device)


args = get_config()
dataname = args.dataname
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

base_name = dataname[4:] if dataname.startswith("pro-") else dataname
pred_ntype = args.predict_ntype_dict[base_name]

print(
    f"[INFO] Loading data: {dataname} | "
    f"Attack: {args.atk_name} | Rate: {args.atk_rate}%"
)

hg, hete_adjs, features, labels, num_classes, train_mask, val_mask, test_mask, pert_data = load_perturbed_data(
    dataname,
    args.atk_name,
    args.atk_rate,
)

hg = hg.to(device)

if isinstance(features, dict):
    features = features[pred_ntype]

features = features.to(device)
labels = labels.to(device)
train_mask = train_mask.bool().to(device)
val_mask = val_mask.bool().to(device)
test_mask = test_mask.bool().to(device)

if pert_data is not None and hasattr(pert_data, "to"):
    pert_data = pert_data.to(device)
    features = pert_data[pred_ntype].x
    labels = pert_data[pred_ntype].y
    train_mask = pert_data[pred_ntype].train_mask.bool()
    val_mask = pert_data[pred_ntype].val_mask.bool()
    test_mask = pert_data[pred_ntype].test_mask.bool()
    hg.nodes[pred_ntype].data["x"] = features


seeds = [42, 43, 44, 45, 46]
all_results = []

os.makedirs("save_model", exist_ok=True)

log_dir = os.path.dirname(args.log_fp)
if log_dir:
    os.makedirs(log_dir, exist_ok=True)


for run_id, seed in enumerate(seeds, start=1):
    current_rate = args.atk_rate

    print(f"\n[Run {run_id}/5] with seed {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.random.seed(seed)

    model = build_model(
        args=args,
        dataname=dataname,
        features=features,
        pred_ntype=pred_ntype,
        num_classes=num_classes,
        device=device,
    )

    stopper = EarlyStopping(patience=args.patience)
    stopper.filename = f"save_model/{args.model}_{dataname}_" + stopper.filename

    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for epoch in trange(args.epochs, desc="Training", leave=False):
        model.train()

        logits = model({pred_ntype: features}, hg)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(
            model,
            hg,
            {pred_ntype: features},
            labels,
            val_mask,
            loss_fcn,
        )

        if stopper.step(val_loss.item(), val_acc, model):
            break

    stopper.load_checkpoint(model)

    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(
        model,
        hg,
        {pred_ntype: features},
        labels,
        test_mask,
        loss_fcn,
    )

    print(
        f"Test Acc: {test_acc:.4f}, "
        f"Micro-F1: {test_micro_f1:.4f}, "
        f"Macro-F1: {test_macro_f1:.4f}"
    )

    final_result = [test_acc, test_micro_f1, test_macro_f1]
    log_result(
        args.log_fp,
        dataname,
        args.atk_name,
        current_rate,
        "PSTABLE",
        final_result,
    )

    all_results.append(final_result)

    del model
    torch.cuda.empty_cache()
    gc.collect()


all_results = np.array(all_results)
avg = all_results.mean(axis=0)
std = all_results.std(axis=0)

print("\nFinal average result over 5 runs:")
print(f"Test Acc:     {avg[0]:.4f} ± {std[0]:.4f}")
print(f"Micro-F1:     {avg[1]:.4f} ± {std[1]:.4f}")
print(f"Macro-F1:     {avg[2]:.4f} ± {std[2]:.4f}")