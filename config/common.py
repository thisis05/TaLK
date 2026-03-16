import os
import random

import numpy as np
import torch

LM_NAME_MAP = {
    "bert": "bert-base-uncased",
    "deberta": "microsoft/deberta-base",
    "roberta": "roberta-base",
}


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_lm_name(short_name: str) -> str:
    if short_name not in LM_NAME_MAP:
        raise ValueError(f"Invalid LM name: {short_name}")
    return LM_NAME_MAP[short_name]
