import argparse


def parse_evaluation_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distilled TAG evaluation")
    parser.add_argument("--gnn_name", type=str, default="GCN")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--cond_size", type=int, default=50)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--gnn_lr", type=float, default=1e-3)
    parser.add_argument("--lm_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lm_name", type=str, default="bert")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--cond_path", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--eval_epoch", type=int, default=500)
    return parser.parse_args()


def apply_evaluation_defaults(args: argparse.Namespace) -> None:
    if args.dataset == "cora":
        args.lm_lr = 1e-4
        args.gnn_lr = 1e-3
        args.inner_early_stopping_patience = 10
        args.early_stopping_patience = 3
    elif args.dataset == "Photo":
        args.lm_lr = 1e-4
        args.gnn_lr = 1e-3
        args.inner_early_stopping_patience = 10
        args.early_stopping_patience = 3
    elif args.dataset == "Computers":
        args.lm_lr = 1e-4
        args.gnn_lr = 1e-3
        args.inner_early_stopping_patience = 8
        args.early_stopping_patience = 3
    elif args.dataset == "arxiv":
        args.lm_lr = 1e-4
        args.gnn_lr = 1e-3
        args.inner_early_stopping_patience = 6
        args.early_stopping_patience = 3
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    return args
