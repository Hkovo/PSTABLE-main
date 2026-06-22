import argparse
import yaml
import os.path as osp

from utils.utils import set_random_seed


predict_ntype_dict = {
    "acm": "paper",
    "dblp": "author",
    "yelp": "b",
}


meta_paths_dict = {
    "acm": [["pa", "ap"], ["pf", "fp"]],
    "dblp": [["ap", "pa"], ["ap", "pc", "cp", "pa"], ["ap", "pt", "tp", "pa"]],
    "yelp": [["bu", "ub"], ["bs", "sb"], ["bl", "lb"]],
}


homo_meta_path_dict = {
    "acm": ["pa", "ap"],
    "dblp": ["ap", "pa"],
    "yelp": ["bs", "sb"],
}


def get_config():
    parser = argparse.ArgumentParser(description="PSTABLE running script.")

    parser.add_argument("--seed", type=int, default=2, help="Random seed.")
    parser.add_argument("--log_fp", type=str, default="results/default.log", help="Log filepath.")
    parser.add_argument(
        "--dataname",
        type=str,
        default="acm",
        choices=["acm", "dblp", "yelp"],
        help="Dataset name.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use.")

    parser.add_argument("--hidden_units", type=int, default=64, help="Number of hidden units.")
    parser.add_argument("--num_heads", type=int, nargs="+", default=[8], help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0.6, help="Dropout rate.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--patience", type=int, default=100, help="Patience for early stopping.")

    parser.add_argument(
        "--atk_name",
        type=str,
        default="HetePRBCD",
        choices=["HetePRBCD", "PRBCD", "FGSM"],
        help="Attack name.",
    )
    parser.add_argument(
        "--atk_rate",
        type=int,
        default=0,
        help="Perturbation percentage.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="PSTABLE",
        choices=["PSTABLE"],
        help="Model name.",
    )

    parser.add_argument("--top_k_ratio", type=float, default=0.05)
    parser.add_argument("--fine_tune_lr", type=float, default=0.001)
    parser.add_argument("--fine_tune_lambda", type=float, default=0.01)
    parser.add_argument("--attack_defense_epochs", type=int, default=100)
    parser.add_argument("--weight_train_epochs", type=int, default=100)
    parser.add_argument("--fine_tune_alpha", type=float, default=0.001)

    args = parser.parse_args()

    cfg_path = f"model_configs/{args.model}.yaml"
    if osp.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if cfg is not None:
            cfg_dict = cfg.get("default", {}) | cfg.get(args.dataname, {})
            parser.set_defaults(**cfg_dict)
            args = parser.parse_args()

    args.meta_paths_dict = meta_paths_dict
    args.homo_meta_path_dict = homo_meta_path_dict
    args.predict_ntype_dict = predict_ntype_dict

    set_random_seed(args.seed)

    return args