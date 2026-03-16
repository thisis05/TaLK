import torch
from torch import nn
from torch.nn import functional as F
from peft import LoraConfig, get_peft_model
import math

def syn_train(G_t, G_s, y_t, y_s, A_t, A_s, train_idx, loss_fn, optimizer, KRR_outer):
    optimizer.zero_grad()

    pred      = KRR_outer.forward(G_t, G_s, y_t, y_s, A_t, A_s)
    y_t_idx   = y_t.argmax(dim=1).to(torch.float32)
    output    = nn.functional.softmax(pred, dim = 1)
    correct   = torch.eq(output[train_idx].argmax(1).to(torch.float32), y_t_idx[train_idx]).sum().item()
    loss      = loss_fn(pred[train_idx], y_t[train_idx])


    loss.backward()
    optimizer.step()

    return loss, correct


def syn_val(G_t, G_s, y_t, y_s, A_t, A_s, val_idx, loss_fn, KRR_outer):
    with torch.no_grad():
        pred = KRR_outer.forward(G_t, G_s, y_t, y_s, A_t, A_s)
        y_t_idx   = y_t.argmax(dim=1).to(torch.float32)
        output      = nn.functional.softmax(pred, dim = 1)
        correct   = torch.eq(output[val_idx].argmax(1).to(torch.float32), y_t_idx[val_idx]).sum().item()
        val_loss = loss_fn(pred[val_idx], y_t[val_idx])

    return val_loss, correct


def syn_test(G_t, G_s, y_t, y_s, A_t, A_s, test_idx, loss_fn, KRR_outer):
    with torch.no_grad():
        pred = KRR_outer.forward(G_t, G_s, y_t, y_s, A_t, A_s)
        y_t_idx   = y_t.argmax(dim=1).to(torch.float32)
        output      = nn.functional.softmax(pred, dim = 1)
        correct   = torch.eq(output[test_idx].argmax(1).to(torch.float32), y_t_idx[test_idx]).sum().item()
        test_loss = loss_fn(pred[test_idx], y_t[test_idx])

    return test_loss, correct

def syn_val_test(G_t, G_s, y_t, y_s, A_t, A_s, val_idx, test_idx, loss_fn, KRR_outer):
    with torch.no_grad():
        pred = KRR_outer.forward(G_t, G_s, y_t, y_s, A_t, A_s)
        y_t_idx   = y_t.argmax(dim=1).to(torch.float32)
        output      = nn.functional.softmax(pred, dim = 1)
        val_correct   = torch.eq(output[val_idx].argmax(1).to(torch.float32), y_t_idx[val_idx]).sum().item()
        test_correct   = torch.eq(output[test_idx].argmax(1).to(torch.float32), y_t_idx[test_idx]).sum().item()
        val_loss = loss_fn(pred[val_idx], y_t[val_idx])

    return val_loss, val_correct, test_correct

def set_mode(lm, mode: str):
    # mode = {'post', 'off'}
    for n, p in lm.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False  
            continue
        if mode == 'post':    
            p.requires_grad = ('.embeddings.' not in n)
        else:                   
            p.requires_grad = False


def build_lora_lm(base_LM, r, alpha, lora_dropout, device, LM_name = "microsoft/deberta-base"):
    
    if LM_name == "microsoft/deberta-base":
        target_modules = ['in_proj', 'pos_proj', 'pos_q_proj']
    elif LM_name in ("bert-base-uncased", "roberta-base"):
        target_modules = ['query', 'key', 'value']
    else:
        raise ValueError(f"Invalid LM name: {LM_name}")
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=lora_dropout,
        bias='none',
        target_modules=target_modules
    )

    lm = get_peft_model(base_LM, lora_cfg).to(device)
    return lm
    
def reset_lora_parameters(lm):
    for module in lm.modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            for adapter_name in module.lora_A.keys():
                lora_A = module.lora_A[adapter_name]
                lora_B = module.lora_B[adapter_name]

                nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(lora_B.weight)

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def snapshot_trainable(model: torch.nn.Module):
    snap = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            snap[n] = p.detach().cpu().clone()
    return snap

def restore_trainable(model: torch.nn.Module, snap: dict):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in snap:
                p.copy_(snap[n].to(p.device))