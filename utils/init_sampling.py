from collections import Counter
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class ClassBalancedSampler:
    def __init__(self, labels, cond_size):
        """
        ClassBalancedSampler that maintains original class distribution proportions.
        
        This sampler:
        1. Counts the number of samples per class in the original dataset
        2. Distributes samples proportionally based on original class distribution
        3. Ensures the total number of sampled indices equals cond_size
        4. The last class gets any remaining samples to ensure exact match
        """
        self.cond_size = cond_size
        self.labels = labels
        self.idx = np.arange(len(labels))

        labels_list = labels.squeeze().tolist()    
        counter = Counter(labels_list)
        self.num_classes = len(counter)
        n_full = len(labels_list)
        
        # Use proportional distribution based on original class distribution (like gcond_base)
        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        sum_ = 0
        self.num_class_dict = {}
        
        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                # Last class: assign remaining samples to ensure exact cond_size
                self.num_class_dict[c] = max(cond_size - sum_, 1)
            else:
                # Proportionally distribute based on original class distribution
                proportion = num / n_full
                self.num_class_dict[c] = max(int(cond_size * proportion), 1)
                sum_ += self.num_class_dict[c]
        
        total = sum(self.num_class_dict.values())
        if total != cond_size:
            diff = cond_size - total
            if diff > 0:
                # Add to the largest class
                largest_class = sorted_counter[-1][0]
                self.num_class_dict[largest_class] += diff
            elif diff < 0:
                # Remove from the largest class
                largest_class = sorted_counter[-1][0]
                self.num_class_dict[largest_class] += diff  # diff is negative
        
        # Print distributions
        orig_dist = {c: num for c, num in sorted(counter.items())}
        syn_dist = {c: num for c, num in sorted(self.num_class_dict.items())}
        print(f"Original distribution (n={n_full}): {orig_dist}")
        print(f"Synthetic distribution (n={cond_size}): {syn_dist}")

    def get_sampled_indices(self):
        selected = []
        for c, k in self.num_class_dict.items():
            idxs = [i for i in range(len(self.labels)) if self.labels[i] == c]
            selected += random.sample(idxs, min(k, len(idxs)))
        if len(selected) < self.cond_size:
            print("Not enough samples for all classes, filling with remaining data")
            print("Random Sampling from other classes")
            pool = set(self.idx) - set(selected)
            needed = self.cond_size - len(selected)
            if pool:
                selected += random.sample(pool, min(needed, len(pool)))

        random.shuffle(selected)
        return selected

class SelectedTextDataset(Dataset):
    def __init__(self, texts, selected_idx, tokenizer, max_length):
        self.texts = texts[selected_idx]    
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return enc['input_ids'].squeeze(0)  # [seq_len]

def sampling(texts, labels, cond_size, tokenizer, max_length, lm, device, save_path = None, return_indices=False):
    cs = ClassBalancedSampler(labels, cond_size)
    selected_idxs = cs.get_sampled_indices()
    sel_dataset = SelectedTextDataset(np.array(texts), selected_idxs, tokenizer, max_length=max_length)
    sel_loader  = DataLoader(
        sel_dataset,
        batch_size=16,    
        shuffle=False,
        pin_memory=True
    )
    
    embed_layer = lm.get_input_embeddings().to(device)
    embed_layer.eval()

    selected_embeddings = []
    with torch.no_grad():
        for batch in tqdm(sel_loader, desc="Generating embeddings"):
            input_ids = batch.to(device)
            embeds = embed_layer(input_ids)  
            selected_embeddings.append(embeds) 

    syn_tokens = torch.cat(selected_embeddings, dim=0).detach().cpu()
    syn_labels = labels[selected_idxs]
    
    # Print actual sampled label distribution
    syn_counter = Counter(syn_labels.squeeze().tolist() if hasattr(syn_labels, 'squeeze') else syn_labels.tolist())
    actual_dist = {c: num for c, num in sorted(syn_counter.items())}
    print(f"Actual sampled distribution (n={len(syn_labels)}): {actual_dist}")
    
    if save_path is not None:
        torch.save(syn_tokens, f'{save_path}/x_s_{cond_size}.pt')
        torch.save(syn_labels, f'{save_path}/y_s_{cond_size}.pt')

    if return_indices:
        return syn_tokens, syn_labels, torch.as_tensor(selected_idxs, dtype=torch.long)
    return syn_tokens, syn_labels
