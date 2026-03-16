import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from GNN import GCNClassifier, SGCClassifier, SAGEClassifier, APPNPClassifier
from config.common import setup_seed
from distill_trainer import (
    build_lora_lm,
    count_trainable,
    reset_lora_parameters,
    restore_trainable,
    set_mode,
    snapshot_trainable,
    syn_val_test,
)
from krr import KRRwithLM_Outer
from sntk import StructureBasedNeuralTangentKernel
from utils.init_sampling import sampling
from utils.load import load_full_data


class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["node_id"] = idx
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def _save_condensed_data(save_dir: str, args, x_s: torch.Tensor, y_s: torch.Tensor, detail: str) -> None:
    x_s_path = os.path.join(save_dir, f"x_s_{detail}_{args.cond_size}.pt")
    y_s_path = os.path.join(save_dir, f"y_s_{detail}_{args.cond_size}.pt")
    torch.save(x_s.detach().cpu(), x_s_path)
    torch.save(y_s.detach().cpu(), y_s_path)
    print(f"Saved condensed data to {save_dir}")

def _get_gnn(gnn_name: str):
    if gnn_name == "GCN":
        from GNN import GCNClassifier as GNNClassifier
    elif gnn_name == "SGC":
        from GNN import SGCClassifier as GNNClassifier
    elif gnn_name == "SAGE":
        from GNN import SAGEClassifier as GNNClassifier
    elif gnn_name == "APPNP":
        from GNN import APPNPClassifier as GNNClassifier
    else:
        raise ValueError(f"GNN Name {gnn_name} is not supported!")
    return GNNClassifier

def run_distillation(args, device: torch.device):
    setup_seed(args.seed)

    data, text, y, y_one_hot, E, idx_train, idx_val, idx_test = load_full_data(
        root=args.data_root,
        name=args.dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        sparse=True,
        seed=args.seed,
    )

    print("Dataset Name:", args.dataset)
    print(f"Seed: {args.seed}")
    print(f"Num nodes: {(len(idx_train) + len(idx_val) + len(idx_test))} (train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)})")
    print(f"condensation size: {args.cond_size} (ratio: {(args.cond_size / len(y)) * 100:.2f}%)")

    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    tokens = tokenizer(text, padding=True, truncation=True, max_length=args.seq_len, return_tensors="pt")
    text_loader = DataLoader(TextDataset(tokens), batch_size=args.batch_size, shuffle=False)

    base_lm = AutoModel.from_pretrained(args.lm_name, use_safetensors=True)
    lm = build_lora_lm(base_lm, args.r, args.alpha, args.lora_dropout, device, args.lm_name)
    reset_lora_parameters(lm)
    print(f"Number of trainable parameters in LM: {count_trainable(lm)}")

    sntk = StructureBasedNeuralTangentKernel(K=args.K, L=args.L, scale=args.scale).to(device)
    kernel = sntk.nodes_gram
    ridge = torch.tensor(args.ridge).to(device)

    if args.syn_init_path is None:
        idx_train_list = list(idx_train)
        texts_candidate = [text[i] for i in idx_train_list]
        labels_candidate = y[idx_train]
        syn_init_path = f"syn_init/{args.dataset}/seed_{args.seed}"
        os.makedirs(syn_init_path, exist_ok=True)
        x_s, y_s = sampling(
            texts_candidate,
            labels_candidate,
            args.cond_size,
            tokenizer,
            args.seq_len,
            base_lm,
            device,
            syn_init_path,
        )
    else:
        x_s = torch.load(f"{args.syn_init_path}/x_s_{args.cond_size}.pt")
        y_s = torch.load(f"{args.syn_init_path}/y_s_{args.cond_size}.pt")

    x_s = x_s.to(device)
    y_s = y_s.to(device)
    y_s = torch.nn.functional.one_hot(y_s.long().view(-1), num_classes=y_one_hot.shape[1]).float()
    x_s = nn.Parameter(x_s, requires_grad=True)

    gnn = _get_gnn(args.gnn_name)(768, args.gnn_hidden_dim, y_s.shape[1], args.gnn_drop_out).to(device)
    syn_loader = DataLoader(torch.utils.data.TensorDataset(x_s, y_s), batch_size=args.batch_size, shuffle=False)

    y = y.to(device)
    idx_train = idx_train.to(device)
    idx_test = idx_test.to(device)
    y_one_hot = y_one_hot.to(device)
    E = E.to(device)

    indices = torch.arange(x_s.shape[0], device=device).unsqueeze(0).repeat(2, 1)
    values = torch.ones(x_s.shape[0], device=device)
    E_s = torch.sparse_coo_tensor(indices, values, size=(x_s.shape[0], x_s.shape[0]), device=device, dtype=torch.float32).coalesce()

    inner_early_stopping_patience = args.inner_early_stopping_patience
    early_stopping_patience = args.early_stopping_patience
    optim_outer = torch.optim.AdamW([x_s], lr=args.distill_lr)

    save_dir = f"./condensed_data/{args.lm_name}/{args.dataset}/"
    os.makedirs(save_dir, exist_ok=True)

    best_eval_acc_val = 0
    best_eval_acc_test = 0
    early_stopping_counter = 0

    for iteration in range(args.iter):
        print("--------------------------------------------------")
        print(f"The {iteration + 1}-th iteration")
        print("--------------------------------------------------")

        for t in range(args.epochs):
            print(f"Epoch {iteration + 1} - {t + 1}")

            gnn = _get_gnn(args.gnn_name)(768, args.gnn_hidden_dim, y_s.shape[1], args.gnn_drop_out).to(device)
            reset_lora_parameters(lm)
            optim_outer = torch.optim.AdamW([x_s], lr=args.distill_lr)

            set_mode(lm, "post")
            lora_post = [p for n, p in lm.named_parameters() if p.requires_grad and "lora_" in n]
            optim_inner = torch.optim.AdamW([
                {"params": lora_post, "lr": args.lm_lr},
                {"params": gnn.parameters(), "lr": args.gnn_lr},
            ])

            lm.train()
            gnn.train()
            inner_early_stopping_counter = 0
            best_train_loss = float("inf")
            best_train_acc = 0
            best_train_lm = None

            pbar = tqdm(range(args.inner_epochs), desc=f"Inner Training - Epoch {t + 1}", leave=False)
            for _ in pbar:
                total_loss = 0.0
                total_acc = 0.0
                for batch_x_s, batch_y_s in syn_loader:
                    optim_inner.zero_grad()
                    batch_x_s = batch_x_s.detach()
                    lm_output = lm(inputs_embeds=batch_x_s)
                    cls_embed = lm_output.last_hidden_state[:, 0, :]

                    batch_e_s = torch.arange(batch_x_s.shape[0], device=device).unsqueeze(0).repeat(2, 1)
                    output = gnn(cls_embed, batch_e_s)
                    loss = F.cross_entropy(output, batch_y_s.argmax(dim=1))
                    loss.backward()
                    optim_inner.step()

                    total_loss += loss.item()
                    total_acc += (output.argmax(dim=1) == batch_y_s.argmax(dim=1)).float().mean().item()

                avg_loss = total_loss / max(1, len(syn_loader))
                avg_acc = total_acc / max(1, len(syn_loader))

                if avg_loss < best_train_loss:
                    best_train_loss = avg_loss
                    best_train_acc = avg_acc
                    best_train_lm = snapshot_trainable(lm)
                    inner_early_stopping_counter = 0
                else:
                    inner_early_stopping_counter += 1
                    if inner_early_stopping_counter >= inner_early_stopping_patience:
                        print(f"Early stopping at epoch {t + 1}")
                        break

                pbar.set_postfix({"Training Loss": avg_loss, "Training Acc": avg_acc, "Early Stopping": inner_early_stopping_counter})

            print(f"[Inner] Best Training Loss: {best_train_loss:.4f} Best Training Acc: {best_train_acc:.4f}")
            if best_train_lm is not None:
                restore_trainable(lm, best_train_lm)

            lm.eval()
            set_mode(lm, "off")
            gnn.eval()

            with torch.inference_mode():
                all_cls_embeds = []
                for batch in tqdm(text_loader, desc=f"Encode real text - Epoch {t + 1}"):
                    outputs = lm(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                    )
                    all_cls_embeds.append(outputs.last_hidden_state[:, 0, :].cpu())
            x_t = torch.cat(all_cls_embeds, dim=0).to(device)

            krr_outer = KRRwithLM_Outer(kernel, ridge, lm).to(device)

            with torch.no_grad():
                eval_output = gnn(x_t, data.edge_index.to(device))
                y_val = y[idx_val]
                y_test = y[idx_test]
                if len(eval_output.shape) == 1:
                    eval_acc_val = (eval_output[idx_val] == y_val).float().mean()
                    eval_acc_test = (eval_output[idx_test] == y_test).float().mean()
                else:
                    eval_acc_val = (eval_output[idx_val].argmax(dim=1) == y_val).float().mean()
                    eval_acc_test = (eval_output[idx_test].argmax(dim=1) == y_test).float().mean()
                print(f"[INFO] LM+GNN Val Acc: {eval_acc_val:.4f}, Test Acc: {eval_acc_test:.4f}")

            if eval_acc_val > best_eval_acc_val:
                best_eval_acc_val = eval_acc_val
                best_eval_acc_test = eval_acc_test
                _save_condensed_data(
                    save_dir,
                    args,
                    x_s,
                    y_s,
                    f"seed_{args.seed}_lr_{args.distill_lr}_outer_{args.outer_epochs}_inner_{args.inner_epochs}",
                )
                print(f"Saved best model at epoch {t + 1}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {t + 1}")
                    break

            for outer_iter in range(args.outer_epochs):
                with torch.inference_mode():
                    all_hs = []
                    for batch_x_s, _ in syn_loader:
                        outputs = lm(inputs_embeds=batch_x_s)
                        all_hs.append(outputs.last_hidden_state[:, 0, :])

                h_s_full = torch.cat(all_hs, dim=0)[: args.cond_size]
                h_s_var = h_s_full.detach().requires_grad_(True)

                indices = torch.arange(args.cond_size, device=device).unsqueeze(0).repeat(2, 1)
                values = torch.ones(args.cond_size, device=device)
                e_s_full = torch.sparse_coo_tensor(indices, values, size=(args.cond_size, args.cond_size), device=device, dtype=torch.float32).coalesce()

                K_ss = kernel(h_s_var, h_s_var, e_s_full, e_s_full)
                K_ts = kernel(x_t, h_s_var, E, e_s_full)

                n_s = h_s_var.shape[0]
                regulizer = ridge * torch.trace(K_ss) * torch.eye(n_s, device=device) / n_s
                b = torch.linalg.solve(K_ss + regulizer, y_s)
                pred = torch.matmul(K_ts, b)

                loss_fn = nn.MSELoss().to(device)
                loss = loss_fn(pred[idx_train], y_one_hot[idx_train])
                loss.backward()
                hs_grads = h_s_var.grad

                avg_loss = loss.item()
                with torch.no_grad():
                    output = F.softmax(pred, dim=1)
                    y_t_idx = y_one_hot.argmax(dim=1).to(torch.float32)
                    avg_correct = torch.eq(output[idx_train].argmax(1).to(torch.float32), y_t_idx[idx_train]).sum().item()
                avg_acc = avg_correct / len(idx_train)

                optim_outer.zero_grad()
                current_idx = 0
                for batch_x_s, _ in syn_loader:
                    batch_size = batch_x_s.shape[0]
                    if current_idx >= args.cond_size:
                        break

                    remaining = min(batch_size, args.cond_size - current_idx)
                    batch_grad_upstream = hs_grads[current_idx : current_idx + remaining]
                    batch_x_s_actual = batch_x_s[:remaining]

                    outputs = lm(inputs_embeds=batch_x_s_actual)
                    batch_hs_new = outputs.last_hidden_state[:, 0, :]
                    batch_hs_new.backward(batch_grad_upstream, retain_graph=False)

                    current_idx += remaining

                optim_outer.step()

                with torch.no_grad():
                    val_loss, val_correct, test_correct = syn_val_test(
                        x_t,
                        x_s,
                        y_one_hot,
                        y_s,
                        E,
                        E_s,
                        idx_val,
                        idx_test,
                        loss_fn,
                        krr_outer,
                    )
                    val_acc = val_correct / len(idx_val)
                    test_acc = test_correct / len(idx_test)

                print(
                    f"[Outer {outer_iter + 1}] Train Acc: {(100 * avg_acc):>0.1f}%, "
                    f"Train loss: {avg_loss:>5f}, Val Acc: {(100 * val_acc):>0.1f}%, "
                    f"Val loss: {val_loss:>5f} Test Acc: {(100 * test_acc):>0.1f}%"
                )

    print(f"--------------- For Seed {args.seed} Train Done! ----------------")
    print(f"For Seed {args.seed} Best Val Acc: {best_eval_acc_val:.4f}")
    print(f"For Seed {args.seed} Best Test Acc: {best_eval_acc_test:.4f}")

    return (
        float(best_eval_acc_val.detach().cpu().numpy()),
        float(best_eval_acc_test.detach().cpu().numpy()),
    )
