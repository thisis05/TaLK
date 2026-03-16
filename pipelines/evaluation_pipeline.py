import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config.common import setup_seed
from distill_trainer import (
    build_lora_lm,
    count_trainable,
    reset_lora_parameters,
    restore_trainable,
    set_mode,
    snapshot_trainable,
)
from utils.init_sampling import sampling
from utils.load import load_full_data


class TextDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["node_id"] = idx
        item["labels"] = self.labels[idx].item()
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


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


def run_evaluation(args, device: torch.device) -> float:
    setup_seed(args.seed)

    data, text, y, y_one_hot, _, idx_train, idx_val, idx_test = load_full_data(
        root=args.data_root,
        name=args.dataset,
        seed=args.seed,
    )

    print("Dataset:", args.dataset)
    print(f"Seed: {args.seed}")
    print(f"Num nodes: {(len(idx_train) + len(idx_val) + len(idx_test))} (train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)})")
    print(f"condensation size: {args.cond_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    tokens = tokenizer(text, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
    text_loader = DataLoader(TextDataset(tokens, y), batch_size=64, shuffle=False)

    base_lm = AutoModel.from_pretrained(args.lm_name, use_safetensors=True)
    lm = build_lora_lm(base_lm, args.r, args.alpha, args.lora_dropout, device, args.lm_name)
    reset_lora_parameters(lm)
    print(f"Number of trainable parameters in LM: {count_trainable(lm)}")

    GNNClassifier = _get_gnn(args.gnn_name)

    x_s = torch.load(
        f"./condensed_data/{args.lm_name}/{args.dataset}/x_s_seed_{args.seed}_{args.cond_path}_{args.cond_size}.pt",
        map_location=torch.device("cpu"),
    ).detach()
    y_s = torch.load(
        f"./condensed_data/{args.lm_name}/{args.dataset}/y_s_seed_{args.seed}_{args.cond_path}_{args.cond_size}.pt",
        map_location=torch.device("cpu"),
    ).detach()

    syn_loader = DataLoader(torch.utils.data.TensorDataset(x_s, y_s), batch_size=args.batch_size, shuffle=True)

    gnn = GNNClassifier(x_s.shape[2], args.hidden_size, y_s.shape[1], args.dropout).to(device)
    set_mode(lm, "post")

    lora_post = [p for n, p in lm.named_parameters() if p.requires_grad and "lora_" in n]
    optimizer = optim.AdamW([
        {"params": lora_post, "lr": args.lm_lr},
        {"params": gnn.parameters(), "lr": args.gnn_lr},
    ])

    y = y.to(device)
    edge_index = data.edge_index.to(device)

    best_val_acc = -float("inf")
    early_stopping_counter = 0
    best_gnn = None
    best_cls_embeds = None

    inner_early_stopping_counter = 0
    best_train_loss = float("inf")
    best_train_acc = 0.0
    best_train_lm = None

    pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in pbar:
        set_mode(lm, "post")
        gnn.train()
        lm.train()

        total_loss = 0.0
        total_acc = 0.0

        for inputs, labels in syn_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)

            lm_output = lm(inputs_embeds=inputs)
            cls_embed = lm_output.last_hidden_state[:, 0, :]

            e_s = torch.arange(cls_embed.size(0), device=device).unsqueeze(0).repeat(2, 1)
            output = gnn(cls_embed, e_s)
            loss = F.cross_entropy(output, labels.argmax(dim=1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += (output.argmax(dim=1) == labels.argmax(dim=1)).float().mean().item()

        avg_loss = total_loss / max(1, len(syn_loader))
        avg_acc = total_acc / max(1, len(syn_loader))
        pbar.set_postfix({"Training Loss": avg_loss, "Training Acc": avg_acc, "Early Stopping": inner_early_stopping_counter})

        if avg_loss < 0.1:
            if avg_loss < best_train_loss:
                best_train_loss = avg_loss
                best_train_acc = avg_acc
                best_train_lm = snapshot_trainable(lm)
                inner_early_stopping_counter = 0
            else:
                inner_early_stopping_counter += 1
                if inner_early_stopping_counter >= args.inner_early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    print(f"[Inner] Best Training Loss: {best_train_loss:.4f} Best Training Acc: {best_train_acc:.4f}")
                    inner_early_stopping_counter = 0

                    if best_train_lm is not None:
                        restore_trainable(lm, best_train_lm)
                    set_mode(lm, "off")
                    lm.eval()
                    gnn.eval()

                    all_cls_embeds = []
                    for batch in tqdm(text_loader, desc=f"Load Embedding - Epoch {epoch + 1}", leave=False):
                        with torch.no_grad():
                            outputs = lm(
                                input_ids=batch["input_ids"].to(device),
                                attention_mask=batch["attention_mask"].to(device),
                            )
                            all_cls_embeds.append(outputs.last_hidden_state[:, 0, :].detach().cpu())

                    cls_embeds = torch.cat(all_cls_embeds, dim=0).to(device)
                    with torch.no_grad():
                        out_all = gnn(cls_embeds, edge_index)
                        pred_val = out_all[idx_val].argmax(dim=1)
                        val_acc = pred_val.eq(y[idx_val]).sum().item() / len(idx_val)

                    print(f"Epoch {epoch + 1}, Validation Acc: {val_acc:.4f}")

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_gnn = {k: v.clone() for k, v in gnn.state_dict().items()}
                        best_cls_embeds = cls_embeds.detach().cpu()
                        early_stopping_counter = 0
                        print(f"[GLOBAL] best_val_acc updated: {best_val_acc:.4f}")
                    else:
                        early_stopping_counter += 1
                        if early_stopping_counter >= args.early_stopping_patience:
                            print(f"Early stopping at epoch {epoch + 1}")
                            break

    if best_gnn is None or best_cls_embeds is None:
        set_mode(lm, "off")
        lm.eval()
        gnn.eval()
        all_cls_embeds = []
        for batch in text_loader:
            with torch.no_grad():
                outputs = lm(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                all_cls_embeds.append(outputs.last_hidden_state[:, 0, :].detach().cpu())
        cls_src = torch.cat(all_cls_embeds, dim=0).to(device)
    else:
        gnn.load_state_dict(best_gnn)
        cls_src = best_cls_embeds.to(device)

    with torch.no_grad():
        out_all = gnn(cls_src, edge_index)
        pred_test = out_all[idx_test].argmax(dim=1)
        test_acc = pred_test.eq(y[idx_test]).sum().item() / len(idx_test)

    print(f"Final Test Acc: {test_acc:.4f}")
    return float(test_acc)
