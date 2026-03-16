import os
import torch

from config.common import resolve_lm_name
from config.distillation_args import apply_distillation_defaults, parse_distillation_args
from pipelines.distillation_pipeline import run_distillation


def main() -> None:
    args = parse_distillation_args()
    apply_distillation_defaults(args)
    args.lm_name = resolve_lm_name(args.lm_name)

    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    val_acc, test_acc = run_distillation(args, device)
    print(f"Seed {args.seed} Val Acc: {val_acc:.4f}")
    print(f"Seed {args.seed} Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
