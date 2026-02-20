# Permutation Equivariant Graph Formulation

This repository contains experiments studying **approximately-equivariant graph neural networks (ASENs)** applied to sequence tasks, traffic forecasting, and human pose estimation.

## Repository Structure

```
perm_equivariance_graph_formulation/
├── SyntheticTaskSeq/          # Multitask & transfer experiments on synthetic invariant/equivariant tasks
│   ├── equiv_experiments.py   # Equivariant (classification) experiments
│   ├── inv_regression.py      # Invariant (regression) experiments
│   ├── data.py                # Dataset definitions
│   └── models/
│       └── Multi_GAT.py       # ASEN model (MultiGraphGATv2)
└── Approx_Equivariant_Graph_Nets/
    ├── DCRNN_Pytorch/         # Traffic forecasting with DCRNN
    └── Human_Pose_Est/        # 3D human pose estimation with equivariant GCN
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **PyTorch Geometric** requires version-matched wheels. See the [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to pick the right `torch_scatter`, `torch_sparse`, etc. for your torch + CUDA version.

---

## SyntheticTaskSeq

Experiments on multi-task learning and transfer learning over synthetic sequence tasks. All scripts are run from the **project root** (not from inside `SyntheticTaskSeq/`).

### Tasks

**Equivariant tasks** (used in `equiv_experiments.py`):

| Task | Description |
|------|-------------|
| `constant` | Detect constant sequences |
| `cyclicaltsum` | Cyclic alternating sum |
| `targetsum` | Target sum detection |
| `monotone` | Monotone sequence detection |
| `cyclicprod` | Cyclic product |
| `symdiff` | Symmetric difference |
| `palindrome` | Palindrome detection |
| `cyclicsum` | Cyclic sum |
| `intersect` | Set intersection |

**Invariant tasks** (used in `inv_regression.py`):

| Task | Description |
|------|-------------|
| `detectcapital` | Detect capital words |
| `vandermonde` | Vandermonde-style regression |
| `cyclicsum` | Cyclic sum regression |
| `longestpal` | Longest palindrome length |
| `palindrome` | Palindrome classification |
| `intersect` | Set intersection regression |

Task universe is ordered by complexity; `--n_task N` selects the **last N** tasks from the list.

---

### `equiv_experiments.py` — Equivariant Classification

#### Functions

| Function | Description |
|----------|-------------|
| `create_datasets(T, n_task, SAMPLE_NUMBER, vocab_size, non_equiv)` | Builds train/val/test splits. `T` selects bundles of 3 tasks; `n_task` selects the last N from the full universe; `non_equiv=True` runs the non-equivariant baseline. |
| `run_multitask_experiments(r, T, bs, trials, single, n_task, same_bs, vocab_size, non_equiv, num_epochs)` | Multitask training: sweeps over task-mix ratios `r` and runs `trials` repeated experiments. `single=True` runs single-task baselines. |
| `run_pretrain_finetune_experiment_equiv(finetune_task, T, n_task, trials, bs, vocab_size, non_equiv, num_pretrain_epochs, num_finetune_epochs)` | Transfer learning: pretrain on all tasks except `finetune_task`, then finetune on `finetune_task`. Compares `pretrain+finetune` vs `finetune_only`. |

#### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `multitask` | `multitask` or `pretrain` |
| `--r` | `0.4` | Data sample ratio (multitask only) |
| `--trials` | `1` | Number of repeated trials |
| `--T` | `1` | Task bundle multiplier (`1`=3 tasks, `2`=6, `3`=9) |
| `--n_task` | `None` | Select last N tasks from the universe |
| `--batch_size` | `64` | Training batch size |
| `--vocab_size` | `7` | Vocabulary size for sequence tasks |
| `--num_epochs` | `40` | Finetune / multitask training epochs |
| `--num_pretrain_epochs` | `15` | Pretrain epochs (`pretrain` mode only) |
| `--finetune_task` | `None` | Task to finetune on (`pretrain` mode); defaults to last loaded task |
| `--single` | `False` | Run single-task baseline |
| `--same_bs` | `False` | Fix batch size across different `n_task` values |
| `--non_equiv` | `False` | Run non-equivariant baseline |
| `--logging` | `False` | Redirect stdout to `SyntheticTaskSeq/logs/` |

#### Examples

```bash
# Multitask with 3-task bundle, 5 trials
python SyntheticTaskSeq/equiv_experiments.py --mode multitask --T 1 --trials 5 --num_epochs 50

# Multitask with 6 tasks (last 6 from universe)
python SyntheticTaskSeq/equiv_experiments.py --mode multitask --n_task 6 --trials 3

# Pretrain on all tasks, finetune on 'intersect', with logging
python SyntheticTaskSeq/equiv_experiments.py --mode pretrain \
    --finetune_task intersect --T 2 --trials 3 \
    --num_pretrain_epochs 20 --num_epochs 40 --logging

# Non-equivariant baseline
python SyntheticTaskSeq/equiv_experiments.py --mode multitask --non_equiv --trials 3
```

---

### `inv_regression.py` — Invariant Regression

#### Functions

| Function | Description |
|----------|-------------|
| `create_datasets(n_task, SAMPLE_NUMBER, vocab_size, non_equiv)` | Builds datasets for invariant regression tasks. `n_task=None` uses the default 4-task set `[longestpal, palindrome, detectcapital, intersect]`; pass an integer to select from the full 6-task universe. |
| `run_experiments_inv(n_task, trials, num_epochs, batch_size, SAMPLE_NUMBER, vocab_size, non_equiv)` | Multitask regression: trains on all loaded tasks simultaneously using L1 regression loss, sweeping over single-task / all-task ratios. |
| `run_pretrain_finetune_experiment(finetune_task, n_task, trials, num_pretrain_epochs, num_finetune_epochs, batch_size, SAMPLE_NUMBER, vocab_size, non_equiv)` | Transfer learning with regression: pretrain with scaled L1 (normalised by output range), then finetune with unscaled L1 on `finetune_task`. Compares `pretrain+finetune` vs `finetune_only`. |
| `run_vandermonde_mlp(trials, num_epochs, batch_size)` | Standalone Vandermonde regression experiment using a permutation-invariant MLP (`PermutationMLP`). |

#### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `multitask` | `multitask`, `pretrain`, or `vandermonde` |
| `--n_task` | `None` | Select last N tasks from the 6-task universe |
| `--trials` | `3` | Number of repeated trials |
| `--batch_size` | `64` | Training batch size |
| `--num_epochs` | `80` | Training / finetune epochs |
| `--num_pretrain_epochs` | `30` | Pretrain epochs (`pretrain` mode only) |
| `--SAMPLE_NUMBER` | `16000` | Data samples per task |
| `--vocab_size` | `10` | Vocabulary size for datasets that support it |
| `--finetune_task` | `None` | Task to finetune on (`pretrain` mode); defaults to last loaded task |
| `--non_equiv` | `False` | Run non-equivariant baseline |
| `--logging` | `False` | Redirect stdout to `SyntheticTaskSeq/logs/` |

#### Examples

```bash
# Multitask over 4 default invariant tasks, 3 trials
python SyntheticTaskSeq/inv_regression.py --mode multitask --trials 3

# Multitask over 6 tasks
python SyntheticTaskSeq/inv_regression.py --mode multitask --n_task 6 --trials 3

# Pretrain → finetune on 'palindrome', with logging
python SyntheticTaskSeq/inv_regression.py --mode pretrain \
    --finetune_task palindrome --n_task 4 --trials 3 \
    --num_pretrain_epochs 30 --num_epochs 50 --logging

# Vandermonde MLP baseline
python SyntheticTaskSeq/inv_regression.py --mode vandermonde --trials 5 --num_epochs 50
```

---

## Approx_Equivariant_Graph_Nets

### DCRNN_Pytorch — Traffic Forecasting

Diffusion Convolutional Recurrent Neural Network adapted with approximately-equivariant graph structure. Run from `Approx_Equivariant_Graph_Nets/DCRNN_Pytorch/`.

#### Entry point

```
dcrnn_train_pytorch.py
```

Reads a YAML config file that specifies dataset paths, model hyperparameters, and training settings.

#### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config_filename` | — | **(required)** Path to YAML config file |
| `--use_cpu_only` | `False` | Force CPU-only execution |
| `--aut` | `False` | Use automorphism-equivariant GNN variant |
| `--orbit_filename` | `orbit_idx.p` | Pickle file containing group orbit/community index lists |

#### Example

```bash
cd Approx_Equivariant_Graph_Nets/DCRNN_Pytorch
python dcrnn_train_pytorch.py \
    --config_filename data/model/dcrnn_la.yaml \
    --orbit_filename orbit_idx.p \
    --aut True
```

The YAML config controls:
- `data.graph_pkl_filename` — path to the sensor adjacency graph pickle
- Model depth, sequence length, learning rate, etc.

---

### Human_Pose_Est — 3D Human Pose Estimation

Equivariant and approximate-equivariant GCN models for lifting 2D keypoints to 3D. Run from `Approx_Equivariant_Graph_Nets/Human_Pose_Est/`.

#### Entry points

| Script | Model |
|--------|-------|
| `main_gcn_equiv.py` | Strictly equivariant SemGCN (`SemGCNEquiv`) |
| `main_gcn_aut.py` | Automorphism-equivariant GCN (`SemGCNAutG`) |
| `main_gatv2.py` | GATv2-based model |
| `main_gat.py` | GAT-based model |
| `main_gine.py` | GINE-based model |
| `main_grit.py` | GRIT-based model |

All scripts share the same CLI interface:

#### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-d` / `--dataset` | `h36m` | Target dataset (only `h36m` supported) |
| `-k` / `--keypoints` | `gt` | 2D keypoint source (`gt` = ground truth) |
| `-a` / `--actions` | `*` | Actions to train/test on (comma-separated or `*` for all) |
| `--evaluate` | `''` | Path to checkpoint to evaluate (skips training) |
| `-r` / `--resume` | `''` | Path to checkpoint to resume training from |
| `-c` / `--checkpoint` | `checkpoint` | Directory to save checkpoints and logs |
| `--snapshot` | `5` | Save checkpoint every N epochs |
| `-l` / `--num_layers` | `4` | Number of residual GCN layers |
| `-z` / `--hid_dim` | `128` | Hidden dimension size |
| `-b` / `--batch_size` | `64` | Batch size (number of poses) |
| `-e` / `--epochs` | `50` | Number of training epochs |
| `--num_workers` | `8` | DataLoader worker threads |
| `--lr` | `1e-3` | Initial learning rate |
| `--lr_decay` | `100000` | Steps between LR decay |
| `--lr_gamma` | `0.96` | LR decay factor |
| `--dropout` | `0.0` | Dropout rate |
| `--downsample` | `1` | Frame rate downsampling factor |

#### Data Setup

Place data files in `Human_Pose_Est/data/`:
- `data_3d_h36m.npz` — 3D ground truth poses
- `data_2d_h36m_gt.npz` — 2D ground truth keypoints

#### Examples

```bash
cd Approx_Equivariant_Graph_Nets/Human_Pose_Est

# Train equivariant model
python main_gcn_equiv.py --epochs 50 --batch_size 64 --checkpoint checkpoints/equiv

# Train automorphism-equivariant model
python main_gcn_aut.py --epochs 50 --checkpoint checkpoints/aut

# Evaluate a saved checkpoint
python main_gcn_equiv.py --evaluate checkpoints/equiv/ckpt_best.pth.tar

# Train on specific actions only
python main_gcn_equiv.py --actions "Walking,Eating,Smoking" --epochs 50
```

Metrics reported: **MPJPE** (Protocol #1) and **P-MPJPE** (Procrustes-aligned, Protocol #2) in millimetres.
