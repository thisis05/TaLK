# Official Code Implementation of TaLK

## Paper Information
* **Title**: TaLK: Text-attributed Graph Dataset Distillation via Coupling Language Model with Graph-Aware Kernel
* **Description**: Official implementation of the dataset distillation algorithm for text-attributed graphs (TAGs) using integrated Language Model (LM) and Graph Neural Network (GNN) learning.

---

## Datasets

The statistics of the datasets used in this work are summarized below:

| Dataset | $\vert V \vert$ | $\vert E \vert$ | # Class |
| :--- | :---: | :---: | :---: |
| **Cora** | 2,708 | 5,429 | 7 |
| **Photo** | 48,362 | 500,928 | 12 |
| **Computers** | 87,229 | 721,081 | 10 |
| **Arxiv** | 169,343 | 1,166,243 | 40 |

* **Cora / Arxiv**: Citation network datasets.
  * The raw text and graph structures are sourced from [here](https://github.com/XiaoxinHe/TAPE).
* **Photo / Computers**: Amazon co-purchase datasets.
  * The raw text and graph structures are sourced from [here](https://github.com/sktsherlock/TAG-Benchmark).

> **Note on Data Preparation:** To download and set up the datasets, please follow the respective instructions provided in the source repositories linked above.

---

## Run Instructions

### 1. Dataset Distillation
To distill a text-attributed graph dataset, run the following command:
```bash
python distillation.py --dataset [cora, Photo, Computers, arxiv] --data_root [ROOT FOLDER] --lm_name [bert, roberta, deberta] --seed [SEED] --cond_size [COND_SIZE] --gpu_id [GPU_ID]
```

### 2. Evaluation on Distilled Datasets
To train and evaluate models using the generated synthetic (distilled) dataset, use:
```bash
python evaluation.py --dataset [cora, Photo, Computers, arxiv] --data_root [ROOT FOLDER] --lm_name [bert, roberta, deberta] --seed [SEED] --cond_size [COND_SIZE] --gpu_id [GPU_ID] --cond_path [PATH FOR THE SAVED SYNTHETIC DATA]
```

---

## Citation
If you find this code or our paper useful in your research, please cite:
```bibtex
[TBD]
```
