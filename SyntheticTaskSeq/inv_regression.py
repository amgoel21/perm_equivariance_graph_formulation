import torch
import pickle
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from models.Multi_GAT import MultiGraphGATv2Model_inv,MultiGraphGATv2Model_equiv, PermutationMLP
from sympy import *
from sympy.combinatorics import Permutation, PermutationGroup
from random import randrange
from data import IsBalancedParenthesisDataset, IsPalindromeDataset, IntersectDataset, MaxCyclicSumDataset, LongestPalindromeDataset, DetectCapitalDataset, Vandermonde
from torch.utils.data import ConcatDataset, DataLoader
from collections import deque, Counter
from sklearn.metrics import confusion_matrix
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_graph_structure(dataset_name, seq_length):
    """
    Creates adjacency matrices and orbit mappings for different dataset structures.
    """

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
    print(len(perms))
    return perms



def create_datasets():
    """
    Creates datasets for multiple graph structures and returns a combined dataset with labels.
    """
    graph_configs = {}
    dataset_splits = {}
    train_data = {}
    test_data={}
    val_data={}

    #structures = ['longestpal','palindrome','intersect','detectcapital']
    structures = ['longestpal','detectcapital','intersect' ,'palindrome']
    total_samples = 4000
    LENGTH_OF_SEQUENCE = 6
    print(LENGTH_OF_SEQUENCE)
    print(total_samples)
    

    for structure_id in structures:
        print(structure_id)
        if structure_id == "palindrome":
            seq_length = LENGTH_OF_SEQUENCE
            dataset = IsPalindromeDataset(num_samples=total_samples, seq_length=seq_length, palindrome_length=3,equivariant=False)
            graph_configs[structure_id] = {
            "n_nodes": seq_length,
            "perms": generate_graph_structure("palindrome", seq_length),
            "coords_dim": (1,1),
            "adj": None,
            "orbits": None,
            "sparse": False,
            "out_dim": 2
        }
        elif structure_id == "intersect":
            seq_length = LENGTH_OF_SEQUENCE
            dataset = IntersectDataset(num_samples=total_samples, seq_length=seq_length, vocab_size=10,equivariant=False, thresh = 4)
            graph_configs[structure_id] = {
            "n_nodes": seq_length,
            "perms": generate_graph_structure("intersect", seq_length),
            "coords_dim": (1,1),
            "adj": None,
            "orbits": None,
            "sparse": False,
            "out_dim": 1+seq_length
        }
        elif structure_id == "cyclicsum":
            seq_length = LENGTH_OF_SEQUENCE
            dataset = MaxCyclicSumDataset(num_samples=total_samples, seq_length=seq_length, cyc_length = 4, vocab_size=10, inv = True)
            graph_configs[structure_id] = {
            "n_nodes": seq_length,
            "perms": generate_graph_structure("cyclicsum", seq_length),
            "coords_dim": (1,1),
            "adj": None,
            "orbits": None,
            "sparse": False,
            "out_dim": seq_length * 4
        }
        elif structure_id == "longestpal":
            seq_length = LENGTH_OF_SEQUENCE
            dataset = LongestPalindromeDataset(num_samples=total_samples, seq_length=seq_length, vocab_size=10, thresh = 5)
            graph_configs[structure_id] = {
            "n_nodes": seq_length,
            "perms": generate_graph_structure("longestpal", seq_length),
            "coords_dim": (1,1),
            "adj": None,
            "orbits": None,
            "sparse": False,
            "out_dim": 1+seq_length
        }
        elif structure_id == "detectcapital":
            seq_length = LENGTH_OF_SEQUENCE
            dataset = DetectCapitalDataset(num_samples=total_samples, word_length=seq_length)
            graph_configs[structure_id] = {
            "n_nodes": seq_length,
            "perms": generate_graph_structure("detectcapital", seq_length),
            "coords_dim": (1,1),
            "adj": None,
            "orbits": None,
            "sparse": False,
            "out_dim": 2
        }
        
        elif structure_id == "vandermonde":
            seq_length = LENGTH_OF_SEQUENCE
            dataset = Vandermonde(num_samples=total_samples, seq_length=seq_length, vocab_size = 10)
            graph_configs[structure_id] = {
            "n_nodes": seq_length,
            "perms": generate_graph_structure("vandermonde", seq_length),
            "coords_dim": (1,1),
            "adj": None,
            "orbits": None,
            "sparse": False,
            "out_dim": 2
        }

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



def run_experiments_inv():
    import matplotlib.pyplot as plt
    import numpy as np

    graph_configs, train_datasets, val_datasets, test_datasets = create_datasets()
    all_structures = list(train_datasets.keys())
    k = len(all_structures)
    if k == 0:
        raise ValueError("No structures found in train_datasets; all_structures is empty.")

    # Use the *smallest* dataset size as the base (as in your original code)
    base_size = min(len(train_datasets[s]) for s in all_structures)

    # Example: must be length k each
    #settings = [[1, 1, 1, 1]]  # <- will error if k != 4
    settings = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

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
        mix_str = " + ".join(f"{frac} {struct}" for frac, struct in zip(setting, all_structures))
        print(f"\nRunning setting: {mix_str}")

        trial_losses = {s: [] for s in all_structures}

        for trial in range(1):
            # Build training subsets according to fractions
            combined_train = []
            for frac, struct in zip(setting, all_structures):
                n = int(base_size * frac)
                if n > 0:
                    combined_train += random.sample(list(train_datasets[struct]), n)

            random.shuffle(combined_train)

            train_loader = DataLoader(
                combined_train, batch_size=64, shuffle=True, collate_fn=collate_fn
            )
            val_loaders = {
                s: DataLoader(
                    val_datasets[s], batch_size=64, shuffle=False, collate_fn=collate_fn
                )
                for s in all_structures
            }

            model = MultiGraphGATv2Model_inv(
                graph_configs, hid_dim=40, num_layers=3, p_dropout=0.1, vocab_size=53
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            num_epochs = 40
            val_epoch_losses = {s: [] for s in all_structures}

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
                print(f"Trial {trial+1}, Epoch {epoch+1}: Train wL1 = {avg_train_loss:.4f}")

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
            print(f"\nValidation losses for Trial {trial+1}:")
            for struct in all_structures:
                print(f"  {struct}: {val_epoch_losses[struct]}")

            # Test
            model.eval()
            with torch.no_grad():
                for s in all_structures:
                    test_loader = DataLoader(
                        test_datasets[s], batch_size=64, shuffle=False, collate_fn=collate_fn
                    )
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

        print("  === Final Avg Test Losses (weighted L1) ===")
        for s in all_structures:
            print(f"    {s}: {np.mean(trial_losses[s]):.4f}")


def run_pretrain_finetune_experiment():
    import numpy as np

    graph_configs, train_datasets, val_datasets, test_datasets = create_datasets()
    all_structures = list(train_datasets.keys())
    assert len(all_structures) >= 1, "No structures found from create_datasets()."

    finetune_task = all_structures[-1]
    pretrain_tasks = [s for s in all_structures if s != finetune_task]
    print(f"Pretraining on: {pretrain_tasks} | Fine-tuning on: {finetune_task}")

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
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(finetune_val, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(finetune_test, batch_size=64, shuffle=False, collate_fn=collate_fn)

    num_pretrain_epochs = 30
    num_finetune_epochs = 50
    print(f"Pretrain Epochs: {num_pretrain_epochs}")
    print(f"Finetune Epochs: {num_finetune_epochs}")

    results = {"pretrain+finetune": [], "finetune_only": []}

    for experiment_type in ["pretrain+finetune", "finetune_only"]:
        print(f"\n====== Starting Experiment: {experiment_type} ======")

        for trial in range(3):
            print(f"\n--- Trial {trial+1} ---")

            model = MultiGraphGATv2Model_inv(
                graph_configs, hid_dim=128, num_layers=4, p_dropout=0.1, vocab_size=53
            ).to(device)

            # L1 everywhere; only pretraining will be range-scaled
            l1_none = torch.nn.L1Loss(reduction='none')
            l1_mean = torch.nn.L1Loss(reduction='mean')

            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # === Pretraining Phase (ONLY for "pretrain+finetune") ===
            if experiment_type == "pretrain+finetune" and len(pretrain_dataset) > 0:
                print("  Pretraining (L1 scaled by 1/(out_dim-1))...")
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
                        print(f"    Pretrain Epoch {epoch+1}: Scaled L1 = {total_loss / max(1, num_batches):.4f}")

            # === Fine-tuning Phase (UNSCALED L1) ===
            print("  Fine-tuning...")
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
            finetune_loader = DataLoader(finetune_train, batch_size=64, shuffle=True, collate_fn=collate_fn)

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
                print(f"Trial {trial+1}, Epoch {epoch+1}: Train L1 = {total_loss / max(1, num_batches):.4f}")

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
                    print(f"    Finetune Epoch {epoch+1}: Val L1 = {avg_val_loss:.4f}")

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
                print(f"  Test L1: {avg_test_loss:.4f}")
            
                # NEW: print per-label averages
                per_label_test = {
                    k: label_err_sum_test[k] / max(1, label_count_test[k])
                    for k in sorted(label_err_sum_test.keys())
                }
                print("  Per-label Test L1:", per_label_test)
            
                results[experiment_type].append(avg_test_loss)


    # === Final results ===
    print("\n====== Final Summary ======")
    for exp_type in results:
        vals = np.array(results[exp_type]) if len(results[exp_type]) else np.array([np.nan])
        print(f"{exp_type}: mean={np.nanmean(vals):.4f} std={np.nanstd(vals):.4f}")



def run_vandermonde_mlp():
    num_trials = 3
    num_epochs = 50
    batch_size = 64

    full_dataset = Vandermonde(num_samples=6000, seq_length=3, vocab_size=15)

    # Print label distribution
    label_counts = Counter([label.item() for _, label in full_dataset])
    print("Label distribution:", label_counts)

    # Split dataset
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

    def collate_fn(batch):
        xs = torch.stack([item[0] for item in batch]).float()  # shape (B, seq_len)
        ys = torch.tensor([item[1] for item in batch])         # shape (B,)
        return xs, ys

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    trial_test_losses = []

    for trial in range(num_trials):
        print(f"\n=== Trial {trial+1} ===")
        model = PermutationMLP(seq_length=3, hidden_dim=128).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

            # Validation
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    preds = model(x_batch)
                    loss = criterion(preds, y_batch)
                    val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
                print(f"           Val Loss = {avg_val_loss:.4f}")

        # Test
        model.eval()
        all_preds, all_labels = [], []
        test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                test_loss += loss.item()

                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        trial_test_losses.append(avg_test_loss)
        print(f"Test Loss: {avg_test_loss:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:")
        print(cm)

    # Summary
    print("\n====== Final Summary ======")
    print(f"Mean Test Loss: {np.mean(trial_test_losses):.4f}")
    print(f"Std Dev:        {np.std(trial_test_losses):.4f}")






if __name__ == "__main__":
    run_pretrain_finetune_experiment()
