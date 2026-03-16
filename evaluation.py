import os
import torch

from config.common import resolve_lm_name
from config.evaluation_args import apply_evaluation_defaults, parse_evaluation_args
from pipelines.evaluation_pipeline import run_evaluation


def main() -> None:
    args = parse_evaluation_args()
    apply_evaluation_defaults(args)
    args.lm_name = resolve_lm_name(args.lm_name)

    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    test_acc = run_evaluation(args, device)
    print(f"Seed {args.seed} Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
