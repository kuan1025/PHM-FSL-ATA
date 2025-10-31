# Few-Shot Plant Pathology Toolkit

This repository contains training, evaluation, and analysis code for few-shot plant disease classification with adversarial task augmentation (ATA). Use the directory guide below to navigate the project structure and locate the relevant scripts, datasets, and utilities.

## Directory and File Guide
| Path | Purpose |
| --- | --- |
| `ATA.py` | Entry point for ATA-enhanced training; wraps meta-learning backbones with adversarial task sampling. |
| `train.py` | Baseline few-shot training script without ATA. |
| `test.py` | Inference script that loads checkpoints, evaluates episodes, and emits logits. |
| `options.py` | Centralised CLI argument parser that shares switches across training and evaluation flows. |
| `utils.py` | Helper utilities used across training and evaluation (logging helpers, metric utilities, etc.). |
| `data/` | Dataset loaders, transforms, and packaged Plant Pathology assets. |
| `data/dataset.py` | Dataset definitions for Plant Pathology 2020/2021 episodes. |
| `data/datamgr.py` | Data manager classes that assemble support/query splits for meta-learning. |
| `data/additional_transforms.py` | Composed torchvision transforms used during augmentation. |
| `data/pp2020`, `data/pp2021` | Image indices (`data.txt`) and image folders consumed by the loaders. |
| `methods/` | Meta-learning method implementations (MatchingNet, ProtoNet, RelationNet, GNN, TPN, shared backbones). |
| `LRPtools/` | Layer-wise relevance propagation helpers for explanation-aware variants of the models. |
| `tools/` | Stand-alone utilities for post-processing and analysis. |
| `tools/logit_collector.py` | Collects logits from saved checkpoints into `.npz` bundles. |
| `tools/metrics_eval.py` | Computes accuracy and other metrics from stored logits. |
| `filelists/` | Episode definitions referencing dataset splits (e.g., Plant Pathology 2020/2021 configurations). |
| `eval_out/` | Default location for evaluation artifacts such as `.npz` logits and metrics summaries. |
| `local/drive/MyDrive/PHM-FSL-ATA` | Checkpoint and experiment output directory matching the sample commands. |


## Usage Examples

### Baseline Training
```bash
python3 train.py \
  --save_dir "./local/drive/MyDrive/PHM-FSL-ATA" \
  --data_dir . \
  --dataset filelists/PP_Field2Field_3way_lg/PlantPathology2020 \
  --train_file train.json \
  --pv_val_file val.json \
  --method GNN --model ResNet10 \
  --train_n_way 3 --test_n_way 3 --n_shot 5 --n_query 8 \
  --name p20_3way_base_r10_gnn \
  --train_aug \
  --eval_every 1 --val_episodes 200 \
  --save_freq 20 \
  --stop_epoch 80 \
  --resume_epoch -1
```

### ATA Training
```bash
python3 ATA.py \
  --save_dir "./local/drive/MyDrive/PHM-FSL-ATA" \
  --data_dir . \
  --dataset filelists/PP_Mix_P20P21_3way_lg/p20_70_p21_30_exact \
  --train_file train.json \
  --pv_val_file val.json \
  --method GNN --model ResNet10 \
  --train_n_way 3 --test_n_way 3 --n_shot 5 --n_query 8 \
  --name p20_70_mix_ATA_r10_gnn_M_lg \
  --train_aug \
  --eval_every 1 --val_episodes 200 \
  --save_freq 20 \
  --stop_epoch 80 \
  --resume_epoch -1 \
  --prob 1.0 --T_max 1 --max_lr 0.5
```

### Evaluation
```bash
python3 tools/metrics_eval.py \
  --pv ./eval_out/p20_3way_base_r10_gnn_lg/pv_val_logits.npz \
  --pd ./eval_out/p20_3way_base_r10_gnn_lg/pd_test_logits.npz \
  --outdir ./eval_out/p20_3way_base_r10_gnn_lg \
  --n_way 3
```

## Acknowledgements

This codebase is based on the work by Wang et al. (2021) and Wang et al. (2023). Please cite the following papers if you use this code:

1. Wang, Haoqing and Deng, Zhi-Hong. (2021). *Cross-Domain Few-Shot Classification via Adversarial Task Augmentation*. Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21. Pages 1075–1081. [DOI: 10.24963/ijcai.2021/149](https://doi.org/10.24963/ijcai.2021/149).
2. Wang, Haoqing, Mai, Huiyu, Gong, Yuhang, and Deng, Zhi-Hong. (2023). *Towards well-generalizing meta-learning via adversarial task augmentation*. *Artificial Intelligence*, 103875. Elsevier.
