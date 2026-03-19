import torch
import pickle
import random
import numpy as np
import argparse
import logging
import os, sys
from torch.utils.data import DataLoader, random_split
from models.Multi_GAT import MultiGraphGATv2Model_inv,MultiGraphGATv2Model_equiv
from sympy import *
from sympy.combinatorics import Permutation, PermutationGroup
from random import randrange
from data import IsBalancedParenthesisDataset, IsPalindromeDataset, IntersectDataset, MaxCyclicSumDataset, LongestPalindromeDataset, DetectCapitalDataset, Vandermonde
from torch.utils.data import ConcatDataset, DataLoader
from collections import deque, Counter
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from plot_utils import plot_training_curves, plot_pretrain_finetune

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_graph_structure(dataset_name, seq_length):
    """
    Creates adjacency matrices and orbit mappings for different dataset structures.
    """
    return [Permutation([i for i in range(seq_length)])]
    if dataset_name == "palindrome":
        # Mirror adjacency (Characters linked to their mirrored counterparts)
        perms = [Permutation([i for i in range(seq_length)]), Permutation([seq_length - i - 1 for i in range(seq_length)])]
        #perms = [Permutation([i for i in range(seq_length)])]
    elif dataset_name == "cyclicsum" or dataset_name == 'cyclicsum2' or dataset_name == 'cyclicsum3':
            identity = [i for i in range(seq_length)]
            perms = [Permutation(identity)]
            oppositeidentity = [seq_length-1-i for i in range(seq_length)]
            perms = [Permutation(identity), Permutation(oppositeidentity)]
            for i in range(seq_length):
                my_list = deque(identity)
                my_list.rotate(i)  # rotate right by 1
                perms.append(Permutation(list(my_list)))
    elif dataset_name == "intersect" or dataset_name == 'setintersect':
        mid = seq_length//2
        identity = [i for i in range(seq_length)]    
        perms=[Permutation(identity)]
        for i in range(1, seq_length):
            if i == mid:
                continue
            perm = identity.copy()  
            perm[i], perm[i-1] = perm[i-1], perm[i]
            perms.append(Permutation(perm))
        perms.append(Permutation([(i + mid) % seq_length for i in range(seq_length)])) 
    elif dataset_name == "set":
        identity = list(range(seq_length))
        perms = [Permutation(identity)]
        for i in range(seq_length - 1):
            perm = identity.copy()
            perm[i], perm[i+1] = perm[i+1], perm[i]  # swap adjacent
            perms.append(Permutation(perm))
    elif dataset_name == 'longestpal':
        identity = [i for i in range(seq_length)]
        perms=[Permutation(identity)]
        for i in range(seq_length - 1):
            perm = identity.copy()  
            perm[i], perm[i+1] = perm[i+1], perm[i]
            perms.append(Permutation(perm))
    elif dataset_name == 'detectcapital':
        identity = [i for i in range(seq_length)]
        perms=[Permutation(identity)]
        for i in range(1,seq_length - 1):
            perm = identity.copy()  
            perm[i], perm[i+1] = perm[i+1], perm[i]
            perms.append(Permutation(perm))
    elif dataset_name == 'vandermonde':
        identity = list(range(seq_length))
        perms = []
        # Generate both (i j k) and (i k j) for i < j < k
        for i in range(seq_length - 2):
            for j in range(i + 1, seq_length - 1):
                for k in range(j + 1, seq_length):
                    # 3-cycle: (i j k)
                    perm1 = identity.copy()
                    perm1[i], perm1[j], perm1[k] = identity[j], identity[k], identity[i]
                    perms.append(Permutation(perm1))
    
                    # 3-cycle: (i k j)
                    perm2 = identity.copy()
                    perm2[i], perm2[j], perm2[k] = identity[k], identity[i], identity[j]
                    perms.append(Permutation(perm2))
        perms.append(Permutation(identity))


    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    logger.info(f"Number of permutations: {len(perms)}")
    return perms



def create_datasets(n_task=None, SAMPLE_NUMBER=16000, vocab_size=10, non_equiv=False):
    """
    Creates datasets for multiple graph structures and returns a combined dataset with labels.
    Args:
    - n_task: select n_task from the task universe; if None, uses the default 4-task set
    - SAMPLE_NUMBER: per-task data samples (70/15/15 train/val/test splits)
    - vocab_size: vocabulary size for datasets that support it
    - non_equiv: if True, run non-equivariant baseline (identity-only graph structure)
    """
    graph_configs = {}
    dataset_splits = {}
    train_data = {}
    test_data = {}
    val_data = {}

    # Full task universe ordered by complexity (select last n_task for experiments)
    if n_task is not None:
        all_structures = ['detectcapital', 'vandermonde', 'cyclicsum', 'longestpal', 'palindrome', 'intersect']
        structures = all_structures[-n_task:]
    else:
        structures = ['longestpal', 'palindrome', 'detectcapital', 'intersect']

    logger.info(f"SAMPLE_NUMBER = {SAMPLE_NUMBER}")
    logger.info(f"vocab_size = {vocab_size}")
    logger.info(f"structures = {structures}")

    for structure_id in structures:
        logger.info(f"Processing structure: {structure_id}")
        seq_length = 8
        if structure_id == "palindrome":
            dataset = IsPalindromeDataset(num_samples=SAMPLE_NUMBER, seq_length=seq_length, palindrome_length=4, equivariant=False)
            graph_configs[structure_id] = {
                "n_nodes": seq_length,
                "perms": generate_graph_structure("palindrome", seq_length),
                "coords_dim": (1, 1),
                "adj": None,
                "orbits": None,
                "sparse": False,
                "out_dim": 2
            }
        elif structure_id == "intersect":
            dataset = IntersectDataset(num_samples=SAMPLE_NUMBER, seq_length=seq_length, vocab_size=vocab_size, equivariant=False, thresh=4)
            graph_configs[structure_id] = {
                "n_nodes": seq_length,
                "perms": generate_graph_structure("intersect", seq_length),
                "coords_dim": (1, 1),
                "adj": None,
                "orbits": None,
                "sparse": False,
                "out_dim": 1 + seq_length
            }
        elif structure_id == "cyclicsum":
            dataset = MaxCyclicSumDataset(num_samples=SAMPLE_NUMBER, seq_length=seq_length, cyc_length=4, vocab_size=vocab_size, inv=True)
            graph_configs[structure_id] = {
                "n_nodes": seq_length,
                "perms": generate_graph_structure("cyclicsum", seq_length),
                "coords_dim": (1, 1),
                "adj": None,
                "orbits": None,
                "sparse": False,
                "out_dim": seq_length * 4
            }
        elif structure_id == "cyclicsum2":
            dataset = MaxCyclicSumDataset(num_samples=SAMPLE_NUMBER, seq_length=seq_length, cyc_length=3, vocab_size=vocab_size, inv=True)
            graph_configs[structure_id] = {
                "n_nodes": seq_length,
                "perms": generate_graph_structure("cyclicsum", seq_length),
                "coords_dim": (1, 1),
                "adj": None,
                "orbits": None,
                "sparse": False,
                "out_dim": seq_length * 3
            }
        elif structure_id == "cyclicsum3":
            dataset = MaxCyclicSumDataset(num_samples=SAMPLE_NUMBER, seq_length=seq_length, cyc_length=5, vocab_size=vocab_size, inv=True)
            graph_configs[structure_id] = {
                "n_nodes": seq_length,
                "perms": generate_graph_structure("cyclicsum", seq_length),
                "coords_dim": (1, 1),
                "adj": None,
                "orbits": None,
                "sparse": False,
                "out_dim": seq_length * 5
            }
        elif structure_id == "longestpal":
            dataset = LongestPalindromeDataset(num_samples=SAMPLE_NUMBER, seq_length=seq_length, vocab_size=vocab_size, thresh=5)
            graph_configs[structure_id] = {
                "n_nodes": seq_length,
                "perms": generate_graph_structure("longestpal", seq_length),
                "coords_dim": (1, 1),
                "adj": None,
                "orbits": None,
                "sparse": False,
                "out_dim": 1 + seq_length
            }
        elif structure_id == "detectcapital":
            dataset = DetectCapitalDataset(num_samples=SAMPLE_NUMBER, word_length=seq_length)
            graph_configs[structure_id] = {
                "n_nodes": seq_length,
                "perms": generate_graph_structure("detectcapital", seq_length),
                "coords_dim": (1, 1),
                "adj": None,
                "orbits": None,
                "sparse": False,
                "out_dim": 2
            }
        elif structure_id == "vandermonde":
            dataset = Vandermonde(num_samples=SAMPLE_NUMBER, seq_length=seq_length, vocab_size=vocab_size)
            graph_configs[structure_id] = {
                "n_nodes": seq_length,
                "perms": generate_graph_structure("vandermonde", seq_length),
                "coords_dim": (1, 1),
                "adj": None,
                "orbits": None,
                "sparse": False,
                "out_dim": 2
            }
        else:
            raise ValueError(f"Unknown structure_id: {structure_id}")

        labeled_dataset = [(seq.unsqueeze(-1), label, structure_id) for seq, label in dataset]

        train_size = int(0.7 * len(labeled_dataset))
        val_size = int(0.15 * len(labeled_dataset))
        test_size = len(labeled_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            labeled_dataset, [train_size, val_size, test_size]
        )
        train_data[structure_id] = train_dataset
        val_data[structure_id] = val_dataset
        test_data[structure_id] = test_dataset

    return graph_configs, train_data, val_data, test_data



def run_experiments_inv(n_task=None, trials=3, num_epochs=80, batch_size=64, SAMPLE_NUMBER=16000, vocab_size=10, non_equiv=False):
    """Multi_task training on invariant tasks with regression. Must preset settings and num datasets"""
    import matplotlib.pyplot as plt
    import numpy as np

    graph_configs, train_datasets, val_datasets, test_datasets = create_datasets(
        n_task=n_task, SAMPLE_NUMBER=SAMPLE_NUMBER, vocab_size=vocab_size, non_equiv=non_equiv
    )
    all_structures = list(train_datasets.keys())
    k = len(all_structures)
    if k == 0:
        raise ValueError("No structures found in train_datasets.")

    # Use the *smallest* dataset size as the base (as in your original code)
    base_size = min(len(train_datasets[s]) for s in all_structures)


    
    settings = [tuple([1 if i == j else 0 for i in range(k)]) for j in range(k)]

    # --- Validate settings shape ---
    if len(settings) == 0:
        raise ValueError("settings must contain at least one list/tuple of mixing fractions.")

    expected_len = k
    first_len = len(settings[0])
    if first_len != expected_len:
        raise ValueError(
            f"Each element of settings must have length {expected_len}, "
            f"but settings[0] has length {first_len}."
        )

    for i, setting in enumerate(settings):
        if len(setting) != expected_len:
            raise ValueError(
                f"settings[{i}] has length {len(setting)}, but expected {expected_len}."
            )

    def collate_fn(batch):
        xs = torch.stack([item[0] for item in batch])  # (B, N, 1)
        ys = torch.tensor([item[1] for item in batch], dtype=torch.float32).unsqueeze(-1)  # (B, 1)
        struct_ids = [item[2] for item in batch]
        return xs, ys, struct_ids

    # Weighted L1 helper (same as before, just untouched)
    def weighted_l1(preds, targets, struct_ids, graph_configs):
        abs_err = (preds - targets).abs()  # (B,1)
        inv_ranges = torch.tensor(
            [1.0 / max(graph_configs[s]["out_dim"] - 1, 1.0) for s in struct_ids],
            dtype=preds.dtype, device=preds.device
        ).unsqueeze(-1)  # (B,1)
        return (abs_err * inv_ranges).mean()

    for setting in settings:
        logger.info(f"Running setting: {setting}")
        trial_losses = {s: [] for s in all_structures}

        for trial in range(trials):
            subsets = []
            for frac, struct in zip(setting, all_structures):
                n = int(base_size * frac)
                if n > 0:
                    combined_train += random.sample(list(train_datasets[struct]), n)

            train_loader = DataLoader(subsets, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            val_loaders = { s: DataLoader(val_datasets[s], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                            for s in all_structures }

            model = MultiGraphGATv2Model_inv(
                graph_configs, hid_dim=40, num_layers=3, p_dropout=0.1, vocab_size=53
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            val_epoch_losses = {s: [] for s in all_structures}
            train_epoch_losses = []

            # Build a general tag string for the mix
            setting_str = "-".join(str(f) for f in setting)


            for epoch in range(num_epochs):
                model.train()
                total_loss = 0.0
                for x_batch, y_batch, struct_ids in train_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    optimizer.zero_grad()
                    preds = model(x_batch, struct_ids)         # (B,1)
                    loss = weighted_l1(preds, y_batch, struct_ids, graph_configs)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_train_loss = total_loss / max(1, len(train_loader))
                train_epoch_losses.append(avg_train_loss)
                logger.info(f"Trial {trial+1}, Epoch {epoch+1}: Train wL1 = {avg_train_loss:.4f}")

                # Validation
                model.eval()
                with torch.no_grad():
                    for struct in all_structures:
                        val_loader = val_loaders[struct]
                        val_loss = 0.0
                        num_batches = 0
                        for x_batch, y_batch, struct_ids in val_loader:
                            x_batch = x_batch.to(device)
                            y_batch = y_batch.to(device)
                            preds = model(x_batch, struct_ids)     # (B,1)
                            loss = weighted_l1(preds, y_batch, struct_ids, graph_configs)
                            val_loss += loss.item()
                            num_batches += 1
                        avg_val_loss = val_loss / max(1, num_batches)
                        val_epoch_losses[struct].append(avg_val_loss)

            # Print validation losses
            logger.info(f"Validation losses for Trial {trial+1}:")
            for struct in all_structures:
                logger.info(f"  {struct}: {val_epoch_losses[struct]}")

            # Test
            model.eval()
            with torch.no_grad():
                for s in all_structures:
                    test_loader = DataLoader(test_datasets[s], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                    test_loss = 0.0
                    num_batches = 0
                    for x_batch, y_batch, struct_ids in test_loader:
                        x_batch = x_batch.to(device)
                        y_batch = y_batch.to(device)
                        preds = model(x_batch, struct_ids)
                        loss = weighted_l1(preds, y_batch, struct_ids, graph_configs)
                        test_loss += loss.item()
                        num_batches += 1
                    avg_test_loss = test_loss / max(1, num_batches)
                    trial_losses[s].append(avg_test_loss)

        logger.info("  === Final Avg Test Losses (weighted L1) ===")
        for s in all_structures:
            logger.info(f"    {s}: {np.mean(trial_losses[s]):.4f}")

    # --- Plot last trial's training curves ---
    if train_epoch_losses:
        plot_training_curves(
            train_losses=train_epoch_losses,
            val_losses_dict=val_epoch_losses if val_epoch_losses else None,
            title="Invariant Multitask Training Curves (last trial)",
            save_path=None,
            show=True,
        )




def run_pretrain_finetune_experiment(finetune_task=None, n_task=None, trials=4, num_pretrain_epochs=30, num_finetune_epochs=50, batch_size=64, SAMPLE_NUMBER=16000, vocab_size=10, non_equiv=False):
    """
    Pretrain on all tasks except finetune_task, then finetune on finetune_task with unscaled L1 regression loss.
    Args:
    - finetune_task: task name to fine-tune on; if None, defaults to the last task in the loaded set
    - n_task, SAMPLE_NUMBER, vocab_size, non_equiv: passed through to create_datasets
    - trials: number of repeated trials
    - num_pretrain_epochs: epochs for the pretraining phase
    - num_finetune_epochs: epochs for the finetuning phase
    - batch_size: batch size for all loaders
    """
    import numpy as np

    graph_configs, train_datasets, val_datasets, test_datasets = create_datasets(
        n_task=n_task, SAMPLE_NUMBER=SAMPLE_NUMBER, vocab_size=vocab_size, non_equiv=non_equiv
    )
    all_structures = list(train_datasets.keys())
    assert len(all_structures) >= 1, "No structures found from create_datasets()."

    # Pick finetune target
    if finetune_task is None:
        finetune_task = all_structures[-1]
    if finetune_task not in all_structures:
        raise ValueError(f"finetune_task '{finetune_task}' not in loaded structures: {all_structures}")
    pretrain_tasks = [s for s in all_structures if s != finetune_task]
    logger.info(f"Pretraining on: {pretrain_tasks} | Fine-tuning on: {finetune_task}")

    # -------- Loaders & helpers (REGRESSION) --------
    def collate_fn(batch):
        xs = torch.stack([item[0] for item in batch])                                   # (B, N, 1)
        ys = torch.tensor([item[1] for item in batch], dtype=torch.float32).unsqueeze(-1)  # (B, 1)
        struct_ids = [item[2] for item in batch]
        return xs, ys, struct_ids

    # Build datasets
    pretrain_dataset = []
    for s in pretrain_tasks:
        pretrain_dataset += list(train_datasets[s])
    random.shuffle(pretrain_dataset)

    finetune_train_full = list(train_datasets[finetune_task])
    finetune_val = val_datasets[finetune_task]
    finetune_test = test_datasets[finetune_task]

    finetune_train_size = max(1, int(0.15 * len(finetune_train_full)))
    finetune_train = random.sample(finetune_train_full, finetune_train_size)

    # DataLoaders
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(finetune_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(finetune_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    logger.info(f"Pretrain Epochs: {num_pretrain_epochs}")
    logger.info(f"Finetune Epochs: {num_finetune_epochs}")

    results = {"pretrain+finetune": [], "finetune_only": []}

    # Track per-epoch losses for plotting (last trial only)
    pretrain_epoch_losses = []
    finetune_epoch_train = []
    finetune_epoch_val = []

    for experiment_type in ["pretrain+finetune", "finetune_only"]:
        logger.info(f"====== Starting Experiment: {experiment_type} ======")

        for trial in range(trials):
            logger.info(f"--- Trial {trial+1} ---")

            model = MultiGraphGATv2Model_inv(
                graph_configs, hid_dim=128, num_layers=4, p_dropout=0.1, vocab_size=53
            ).to(device)

            # L1 everywhere; only pretraining will be range-scaled
            l1_none = torch.nn.L1Loss(reduction='none')
            l1_mean = torch.nn.L1Loss(reduction='mean')

            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # === Pretraining Phase (ONLY for "pretrain+finetune") ===
            pretrain_epoch_losses = []
            finetune_epoch_train = []
            finetune_epoch_val = []
            if experiment_type == "pretrain+finetune" and len(pretrain_dataset) > 0:
                logger.info("  Pretraining (L1 scaled by 1/(out_dim-1))...")
                for epoch in range(num_pretrain_epochs):
                    model.train()
                    total_loss = 0.0
                    num_batches = 0
                    for x_batch, y_batch, struct_ids in pretrain_loader:
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                        optimizer.zero_grad()
                        preds = model(x_batch, struct_ids)  # (B,1)

                        # Per-sample |error|
                        abs_err = l1_none(preds, y_batch)   # (B,1)

                        # Scale by 1 / (out_dim - 1) per task
                        inv_ranges = torch.tensor(
                            [1.0 / max(graph_configs[s]["out_dim"] - 1, 1e-8) for s in struct_ids],
                            dtype=torch.float32, device=preds.device
                        ).unsqueeze(-1)  # (B,1)

                        loss = (abs_err * inv_ranges).mean()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        num_batches += 1

                    if (epoch + 1) % 5 == 0:
                        avg_pt_loss = total_loss / max(1, num_batches)
                        pretrain_epoch_losses.append(avg_pt_loss)
                        logger.info(f"    Pretrain Epoch {epoch+1}: Scaled L1 = {avg_pt_loss:.4f}")

            # === Fine-tuning Phase (UNSCALED L1) ===
            logger.info("  Fine-tuning...")
            gnn_params, edge_embedder_params = [], []
            for name, param in model.named_parameters():
                if "edge_embedders" in name:
                    edge_embedder_params.append(param)
                else:
                    gnn_params.append(param)

            optimizer = torch.optim.Adam([
                {'params': gnn_params, 'lr': 0.001},       # smaller LR for core GNN
                {'params': edge_embedder_params, 'lr': 0.02}  # larger LR for edge embedders
            ])
            finetune_loader = DataLoader(finetune_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

            val_losses = []
            for epoch in range(num_finetune_epochs):
                model.train()
                total_loss = 0.0
                num_batches = 0
                for x_batch, y_batch, struct_ids in finetune_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    preds = model(x_batch, struct_ids)  # (B,1)
                    loss = l1_mean(preds, y_batch)      # unscaled during fine-tune
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    num_batches += 1
                ft_train_loss = total_loss / max(1, num_batches)
                finetune_epoch_train.append(ft_train_loss)
                logger.info(f"Trial {trial+1}, Epoch {epoch+1}: Train L1 = {ft_train_loss:.4f}")

                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    num_batches = 0
                    for x_batch, y_batch, struct_ids in val_loader:
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                        preds = model(x_batch, struct_ids)  # (B,1)
                        loss = l1_mean(preds, y_batch)      # unscaled val
                        val_loss += loss.item()
                        num_batches += 1
                    avg_val_loss = val_loss / max(1, num_batches)
                    val_losses.append(avg_val_loss)
                    finetune_epoch_val.append(avg_val_loss)
                    logger.info(f"    Finetune Epoch {epoch+1}: Val L1 = {avg_val_loss:.4f}")

            # # === Test Phase (UNSCALED L1) ===
            # model.eval()
            # with torch.no_grad():
            #     test_loss = 0.0
            #     num_batches = 0
            #     for x_batch, y_batch, struct_ids in test_loader:
            #         x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            #         preds = model(x_batch, struct_ids)  # (B,1)
            #         loss = l1_mean(preds, y_batch)
            #         test_loss += loss.item()
            #         num_batches += 1
            #     avg_test_loss = test_loss / max(1, num_batches)
            #     print(f"  Test L1: {avg_test_loss:.4f}")
            #     results[experiment_type].append(avg_test_loss)
            # === Test Phase (UNSCALED L1) ===
            
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                num_batches = 0
            
                # NEW: per-label accumulators
                label_err_sum_test = defaultdict(float)
                label_count_test = defaultdict(int)
            
                for x_batch, y_batch, struct_ids in test_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    preds = model(x_batch, struct_ids)  # (B,1)
            
                    # Overall unscaled L1 for reporting consistency
                    loss = l1_mean(preds, y_batch)
                    test_loss += loss.item()
                    num_batches += 1
            
                    # --- NEW: per-label L1 accumulation ---
                    abs_err = (preds - y_batch).abs().squeeze(-1)         # (B,)
                    labels_rounded = y_batch.squeeze(-1).round().long()   # (B,)
                    # If your labels are already exact integers (0/1/etc.), you can skip .round()
            
                    # Accumulate sums and counts per label
                    for lbl in labels_rounded.unique():
                        mask = (labels_rounded == lbl)
                        label_err_sum_test[int(lbl.item())] += abs_err[mask].sum().item()
                        label_count_test[int(lbl.item())] += int(mask.sum().item())
            
                avg_test_loss = test_loss / max(1, num_batches)
                logger.info(f"  Test L1: {avg_test_loss:.4f}")
            
                # NEW: print per-label averages
                per_label_test = {
                    k: label_err_sum_test[k] / max(1, label_count_test[k])
                    for k in sorted(label_err_sum_test.keys())
                }
                logger.info(f"  Per-label Test L1: {per_label_test}")
            
                results[experiment_type].append(avg_test_loss)


    # === Final results ===
    logger.info("====== Final Summary ======")
    for exp_type in results:
        vals = np.array(results[exp_type]) if len(results[exp_type]) else np.array([np.nan])
        logger.info(f"{exp_type}: mean={np.nanmean(vals):.4f} std={np.nanstd(vals):.4f}")

    # --- Plot last trial's pretrain+finetune curves ---
    if pretrain_epoch_losses or finetune_epoch_train:
        plot_pretrain_finetune(
            pretrain_losses=pretrain_epoch_losses,
            finetune_train_losses=finetune_epoch_train,
            finetune_val_losses=finetune_epoch_val or None,
            title=f"Pretrain → Finetune ({finetune_task})",
            save_path=None,
            show=True,
        )



# def run_vandermonde_mlp(trials=3, num_epochs=50, batch_size=64):
#     """Vandermonde tests, not for general invariance tests"""

#     full_dataset = Vandermonde(num_samples=6000, seq_length=3, vocab_size=15)

#     # Print label distribution
#     label_counts = Counter([label.item() for _, label in full_dataset])
#     logger.info(f"Label distribution: {label_counts}")

#     # Split dataset
#     train_size = int(0.7 * len(full_dataset))
#     val_size = int(0.15 * len(full_dataset))
#     test_size = len(full_dataset) - train_size - val_size
#     train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

#     def collate_fn(batch):
#         xs = torch.stack([item[0] for item in batch]).float()  # shape (B, seq_len)
#         ys = torch.tensor([item[1] for item in batch])         # shape (B,)
#         return xs, ys

#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
#     val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

#     trial_test_losses = []

#     for trial in range(trials):
#         logger.info(f"=== Trial {trial+1} ===")
#         model = PermutationMLP(seq_length=3, hidden_dim=128).to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#         criterion = torch.nn.CrossEntropyLoss()

#         for epoch in range(num_epochs):
#             model.train()
#             total_loss = 0.0
#             for x_batch, y_batch in train_loader:
#                 x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#                 optimizer.zero_grad()
#                 preds = model(x_batch)
#                 loss = criterion(preds, y_batch)
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()
#             avg_train_loss = total_loss / len(train_loader)
#             logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

#             # Validation
#             model.eval()
#             with torch.no_grad():
#                 val_loss = 0.0
#                 for x_batch, y_batch in val_loader:
#                     x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#                     preds = model(x_batch)
#                     loss = criterion(preds, y_batch)
#                     val_loss += loss.item()
#                 avg_val_loss = val_loss / len(val_loader)
#                 logger.info(f"           Val Loss = {avg_val_loss:.4f}")

#         # Test
#         model.eval()
#         all_preds, all_labels = [], []
#         test_loss = 0.0
#         with torch.no_grad():
#             for x_batch, y_batch in test_loader:
#                 x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#                 logits = model(x_batch)
#                 loss = criterion(logits, y_batch)
#                 test_loss += loss.item()

#                 preds = logits.argmax(dim=1).cpu().numpy()
#                 all_preds.extend(preds)
#                 all_labels.extend(y_batch.cpu().numpy())

#         avg_test_loss = test_loss / len(test_loader)
#         trial_test_losses.append(avg_test_loss)
#         logger.info(f"Test Loss: {avg_test_loss:.4f}")

#         # Confusion Matrix
#         cm = confusion_matrix(all_labels, all_preds)
#         logger.info(f"Confusion Matrix:\n{cm}")

#     # Summary
#     logger.info("====== Final Summary ======")
#     logger.info(f"Mean Test Loss: {np.mean(trial_test_losses):.4f}")
#     logger.info(f"Std Dev:        {np.std(trial_test_losses):.4f}")






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="multitask",
                        choices=["multitask", "pretrain"],
                        help='Which experiment to run')
    parser.add_argument('--n_task', type=int, default=None,
                        help='Number of tasks to select from the task universe (selects last n)')
    parser.add_argument('--trials', type=int, default=3,
                        help='Number of experiment trials')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=80,
                        help='Number of training/finetune epochs')
    parser.add_argument('--num_pretrain_epochs', type=int, default=30,
                        help='Number of pretraining epochs (pretrain mode only)')
    parser.add_argument('--SAMPLE_NUMBER', type=int, default=16000,
                        help='Data samples per task')
    parser.add_argument('--vocab_size', type=int, default=10,
                        help='Vocabulary size for datasets that support it')
    parser.add_argument('--finetune_task', type=str, default=None,
                        help='Task to fine-tune on (pretrain mode); defaults to last task in loaded set')
    parser.add_argument('--non_equiv', action='store_true',
                        help='Run non-equivariant baseline (identity-only graph structure)')
    parser.add_argument('--logging', action='store_true',
                        help='Also log to a file in SyntheticTaskSeq/logs/')
    args = parser.parse_args()

    # --- Configure logging ---
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]

    if args.logging:
        ntask_str = f"n={args.n_task}" if args.n_task is not None else "default"
        if args.mode == "pretrain":
            ft = args.finetune_task if args.finetune_task else "default"
            name = f"pretrain_ft={ft}_{ntask_str}_trials={args.trials}_vocab={args.vocab_size}_nonEquiv={args.non_equiv}"
        else:
            name = f"{args.mode}_{ntask_str}_trials={args.trials}_vocab={args.vocab_size}_nonEquiv={args.non_equiv}"
        save_path = f"SyntheticTaskSeq/logs/{name}"
        os.makedirs(save_path, exist_ok=True)
        log_file_path = os.path.join(save_path, f"experiment_samples={args.SAMPLE_NUMBER}_bs={args.batch_size}.log")
        handlers.append(logging.FileHandler(log_file_path, mode='w'))

    logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers)

    if args.mode == "multitask":
        run_experiments_inv(
            n_task=args.n_task,
            trials=args.trials,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            SAMPLE_NUMBER=args.SAMPLE_NUMBER,
            vocab_size=args.vocab_size,
            non_equiv=args.non_equiv,
        )
    elif args.mode == "pretrain":
        run_pretrain_finetune_experiment(
            finetune_task=args.finetune_task,
            n_task=args.n_task,
            trials=args.trials,
            num_pretrain_epochs=args.num_pretrain_epochs,
            num_finetune_epochs=args.num_epochs,
            batch_size=args.batch_size,
            SAMPLE_NUMBER=args.SAMPLE_NUMBER,
            vocab_size=args.vocab_size,
            non_equiv=args.non_equiv,
        )
    elif args.mode == "vandermonde":
        run_vandermonde_mlp(
            trials=args.trials,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
        )

    if args.logging:
        logger.info(f"Logs written to: {log_file_path}")
