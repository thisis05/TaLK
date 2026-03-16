# TaLK
Text-attributed Graph Dataset Distillation via Coupling Language Model with Graph-aware Kernel

### Distillation 
```python distillation.py --dataset [cora, Photo, Computers, arxiv] --data_root [ROOT FOLDER] --lm_name [bert, roberta, deberta]  --seed [SEED]  --cond_size [COND_SIZE]  --gpu_id [GPU_ID]```

### Evaluation 
```python evaluation.py --dataset [cora, Photo, Computers, arxiv] --data_root [ROOT FOLDER] --lm_name [bert, roberta, deberta] --seed [SEED] --cond_size [COND_SIZE] --gpu_id  [GPU_ID] --cond_path [PATH FOR THE SAVED SYNTHETIC DATA]```
