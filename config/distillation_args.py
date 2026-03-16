import argparse


def parse_distillation_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TAG distillation")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--cond_size", type=int, default=40)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--outer_epochs", type=int, default=10)
    parser.add_argument("--inner_epochs", type=int, default=500)
    parser.add_argument("--iter", type=int, default=1)

    parser.add_argument("--lm_lr", type=float, default=1e-4)
    parser.add_argument("--distill_lr", type=float, default=1e-4)
    parser.add_argument("--gnn_lr", type=float, default=1e-4)

    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--scale", type=str, default="average")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--lm_name", type=str, default="bert")
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--early_stopping_patience", type=int, default=10)

    parser.add_argument("--gnn_name", type=str, default="GCN")
    parser.add_argument("--gnn_hidden_dim", type=int, default=128)
    parser.add_argument("--gnn_drop_out", type=float, default=0.3)
    parser.add_argument("--syn_init_path", type=str, default=None)
    return parser.parse_args()


def apply_distillation_defaults(args: argparse.Namespace) -> None:
    if args.dataset == "cora":
        args.distill_lr = 5e-4
        args.outer_epochs = 8
        args.lm_lr = 1e-4
        args.gnn_lr = 1e-3
        args.ridge = 1e-4
        args.gnn_hidden_dim = 256
        args.inner_early_stopping_patience = 10
        args.inner_epochs = 90
        args.epochs = 80
        args.K = 2
        args.L = 1
    elif args.dataset == "Computers":
        args.distill_lr = 1e-4
        args.outer_epochs = 8
        args.lm_lr = 1e-4
        args.gnn_lr = 1e-3
        args.ridge = 1e-5
        args.gnn_hidden_dim = 256
        args.inner_epochs = 80
        args.K = 2
        args.L = 1
        args.inner_early_stopping_patience = 8
    elif args.dataset == "Photo":
        args.distill_lr = 1e-4
        args.outer_epochs = 8
        args.lm_lr = 1e-4
        args.gnn_lr = 1e-3
        args.ridge = 1e-5
        args.gnn_hidden_dim = 256
        args.inner_epochs = 60
        args.K = 2
        args.L = 1
        args.inner_early_stopping_patience = 10
    elif args.dataset == "arxiv":
        args.distill_lr = 1e-4
        args.outer_epochs = 5
        args.lm_lr = 1e-4
        args.gnn_lr = 1e-3
        args.ridge = 1e-5
        args.gnn_hidden_dim = 256
        args.K = 2
        args.L = 1
        args.inner_early_stopping_patience = 6
        args.inner_epochs = 60
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    return args