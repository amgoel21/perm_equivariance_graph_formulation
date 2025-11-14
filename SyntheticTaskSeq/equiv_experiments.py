import torch
import pickle
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from models.Multi_GAT import MultiGraphGATv2Model_inv,MultiGraphGATv2Model_equiv
from sympy import *
from sympy.combinatorics import Permutation, PermutationGroup
from random import randrange
from data import IsBalancedParenthesisDataset, IsPalindromeDataset, IntersectDataset, MaxCyclicSumDataset, SetGameDataset, SETIntersect  # Import dataset classes
from torch.utils.data import ConcatDataset, DataLoader
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_graph_structure(dataset_name, seq_length):
    """
    Creates adjacency matrices and orbit mappings for different dataset structures.
    """

    if dataset_name == "palindrome":
        perms = [Permutation([i for i in range(seq_length)]), Permutation([seq_length - i - 1 for i in range(seq_length)])]
        #perms = [Permutation([i for i in range(seq_length)])]
    elif dataset_name == "cyclicsum":
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

    # Define structure names (or structure IDs)
    structures = [ "palindrome","cyclicsum", "intersect"]
    LENGTH_OF_SEQUENCE = 10
    SAMPLE_NUMBER = 2500
    print(LENGTH_OF_SEQUENCE)
    print(SAMPLE_NUMBER)
    for structure_id in structures:
        print(structure_id)
        if structure_id == "palindrome":
            seq_length = LENGTH_OF_SEQUENCE
            dataset = IsPalindromeDataset(num_samples=SAMPLE_NUMBER, seq_length=seq_length, palindrome_length=4,equivariant=True)
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
            dataset = IntersectDataset(num_samples=SAMPLE_NUMBER, seq_length=seq_length, vocab_size=13,equivariant=True)
            graph_configs[structure_id] = {
            "n_nodes": seq_length,
            "perms": generate_graph_structure("intersect", seq_length),
            "coords_dim": (1,1),
            "adj": None,
            "orbits": None,
            "sparse": False,
            "out_dim": 4
        }
        elif structure_id == "cyclicsum":
            seq_length = LENGTH_OF_SEQUENCE
            dataset = MaxCyclicSumDataset(num_samples=SAMPLE_NUMBER, seq_length=seq_length, cyc_length = 4, vocab_size=13)
            graph_configs[structure_id] = {
            "n_nodes": seq_length,
            "perms": generate_graph_structure("cyclicsum", seq_length),
            "coords_dim": (1,1),
            "adj": None,
            "orbits": None,
            "sparse": False,
            "out_dim": 4
        }
        elif structure_id == "set":
            seq_length = LENGTH_OF_SEQUENCE
            dataset = SetGameDataset(num_samples=SAMPLE_NUMBER, seq_length=seq_length)
            graph_configs[structure_id] = {
            "n_nodes": seq_length,
            "perms": generate_graph_structure("set", seq_length),
            "coords_dim": (1,1),
            "adj": None,
            "orbits": None,
            "sparse": False,
            "out_dim": 4
        }
        elif structure_id == "setintersect":
            seq_length = LENGTH_OF_SEQUENCE
            dataset = SETIntersect(num_samples=SAMPLE_NUMBER, seq_length=seq_length)
            graph_configs[structure_id] = {
            "n_nodes": seq_length,
            "perms": generate_graph_structure("setintersect", seq_length),
            "coords_dim": (1,1),
            "adj": None,
            "orbits": None,
            "sparse": False,
            "out_dim": 4
        }

        labeled_dataset = [(seq.unsqueeze(-1), label, structure_id) for seq, label in dataset]

        # # Split into train/test
        # train_size = int(0.8 * len(labeled_dataset))
        # test_size = len(labeled_dataset) - train_size
        # train_dataset, test_dataset = random_split(labeled_dataset, [train_size, test_size])
        # train_data[structure_id] = train_dataset
        # test_data[structure_id] = test_dataset

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

@torch.no_grad()
def dump_edge_features(model, graph_configs, out_dir: str, tag: str = ""):
    """
    Robustly dump learned edge feature vectors for *every ordered node pair (i,j)*
    for each structure in `graph_configs`.

    Accepts graph_configs values as either:
      - dataclass with attribute `n_nodes`, or
      - dict with key 'n_nodes'.

    Expected model API:
        model.edge_features_for_pairs(structure_id: str,
                                      pairs: torch.LongTensor [N,2])
            -> Dict[str, torch.Tensor [N,D]]  OR  torch.Tensor [N,D]
    """
    import os, json
    import numpy as np
    import torch

    os.makedirs(out_dir, exist_ok=True)

    # Safe device resolution even if the model has no registered params
    try:
        device = next(p.device for p in model.parameters())
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for struct_id, cfg in graph_configs.items():
        # --- support dataclass or plain dict ---
        if isinstance(cfg, dict):
            if "n_nodes" not in cfg:
                raise KeyError(f"graph_configs['{struct_id}'] has no 'n_nodes' key")
            n = int(cfg["n_nodes"])
        else:
            # dataclass or object with attribute
            if not hasattr(cfg, "n_nodes"):
                raise AttributeError(f"graph_configs['{struct_id}'] has no attribute n_nodes")
            n = int(getattr(cfg, "n_nodes"))
        # ---------------------------------------

        # all ordered pairs (i,j), including self-edges
        I, J = torch.meshgrid(
            torch.arange(n, device=device),
            torch.arange(n, device=device),
            indexing="ij",
        )
        pairs = torch.stack([I.reshape(-1), J.reshape(-1)], dim=-1)  # [N,2]

        # Call into your model (must be implemented)
        feat_out = model.edge_features_for_pairs(struct_id, pairs)

        # normalize to dict[str -> tensor]
        if isinstance(feat_out, torch.Tensor):
            feat_dict = {"edge": feat_out}
        elif isinstance(feat_out, dict):
            feat_dict = feat_out
        else:
            raise TypeError(
                f"edge_features_for_pairs must return a Tensor or Dict[str, Tensor]; got {type(feat_out)}"
            )

        np_pairs = pairs.detach().cpu().numpy()
        save_dict = {"pairs": np_pairs}
        layer_names = []

        for k, v in feat_dict.items():
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"Feature '{k}' is not a Tensor (got {type(v)})")
            if v.shape[0] != np_pairs.shape[0]:
                raise ValueError(
                    f"Feature '{k}' first dim mismatch: {v.shape[0]} vs {np_pairs.shape[0]}"
                )
            layer_names.append(k)
            save_dict[k] = v.detach().cpu().numpy()

        tag_sfx = f"__{tag}" if tag else ""
        base = os.path.join(out_dir, f"edge_feats__{struct_id}{tag_sfx}")

        np.savez_compressed(base + ".npz", **save_dict)
        with open(base + ".meta.json", "w") as f:
            json.dump(
                {
                    "structure": struct_id,
                    "n_nodes": n,
                    "num_pairs": int(np_pairs.shape[0]),
                    "layers": layer_names,
                    "arrays": list(save_dict.keys()),  # includes 'pairs'
                },
                f,
                indent=2,
            )

        print(f"[dump_edge_features] wrote {base}.npz (pairs={np_pairs.shape[0]}, layers={layer_names})")





def run_multitask_experiments():
    import copy
    import matplotlib.pyplot as plt
    import numpy as np

    graph_configs, train_datasets, val_datasets, test_datasets = create_datasets()
    all_structures = list(train_datasets.keys())
    k = len(all_structures)
    if k == 0:
        raise ValueError("No structures found in train_datasets; all_structures is empty.")


    # Example settings (must be length k each):
    # settings = [
    #     (1.0, 1.0, 1.0),
    #     (1.0, 0.2, 0.2),
    #     (0.2, 1.0, 0.2),
    # ]
    settings = [(1, 0, 0), (0, 0, 1)]  # <-- must match k; will error if k != 3

    # --- Validate settings shape ---
    if len(settings) == 0:
        raise ValueError("settings must contain at least one tuple of mixing fractions.")

    expected_len = k
    first_len = len(settings[0])
    if first_len != expected_len:
        raise ValueError(
            f"Each element of settings must have length {expected_len}, "
            f"but settings[0] has length {first_len}."
        )

    for i, frac_tuple in enumerate(settings):
        if len(frac_tuple) != expected_len:
            raise ValueError(
                f"settings[{i}] has length {len(frac_tuple)}, but expected {expected_len}."
            )

    def collate_fn(batch):
        xs = torch.stack([item[0] for item in batch])  # (B, ...)
        ys = torch.stack([item[1] for item in batch])  # (B, N)
        struct_ids = [item[2] for item in batch]       # list of length B
        return xs, ys, struct_ids

    results = {}

    # Use the size of the first structure as base size (as before)
    base_size = len(train_datasets[all_structures[0]])

    for frac_tuple in settings:
        # Pretty printing for arbitrary k
        mix_str = " + ".join(
            f"{frac} {struct}"
            for frac, struct in zip(frac_tuple, all_structures)
        )
        print(f"\nRunning setting: {mix_str}")

        # Track test losses per structure over trials
        trial_losses = {struct: [] for struct in all_structures}

        # Build combined_train for this setting
        n_total = int(base_size)
        subsets = []
        for struct, frac in zip(all_structures, frac_tuple):
            n = int(n_total * frac)
            subset = random.sample(list(train_datasets[struct]), n) if n > 0 else []
            subsets.append(subset)
        combined_train = [item for subset in subsets for item in subset]

        for trial in range(1):
            # print(f"  Trial {trial + 1}/3")
            random.shuffle(combined_train)

            train_loader = DataLoader(
                combined_train,
                batch_size=64,
                shuffle=True,
                collate_fn=collate_fn
            )

            val_loaders = {
                struct: DataLoader(
                    list(val_datasets[struct]),
                    batch_size=64,
                    shuffle=False,
                    collate_fn=collate_fn
                )
                for struct in all_structures
            }

            model = MultiGraphGATv2Model_equiv(
                graph_configs, hid_dim=128, num_layers=4,
                p_dropout=0.1, vocab_size=81
            ).to(device)

            gnn_params = []
            edge_embedder_params = []
            token_embedder_params = []

            for name, param in model.named_parameters():
                if "edge_embedders" in name:
                    edge_embedder_params.append(param)
                elif "token_embedders" in name:
                    token_embedder_params.append(param)
                else:
                    gnn_params.append(param)

            optimizer = torch.optim.Adam([
                {'params': gnn_params,          'lr': 0.01},  # GNN body
                {'params': edge_embedder_params,'lr': 0.03},  # Edge embedders
                {'params': token_embedder_params,'lr': 0.1},  # Token embedders
            ])
            criterion = torch.nn.CrossEntropyLoss()
            num_epochs = 40

            train_epoch_losses = []
            val_epoch_losses = {struct: [] for struct in all_structures}

            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0.0

                for batch_inputs, batch_targets, structure_ids in train_loader:
                    batch_inputs = batch_inputs.to(device)
                    batch_targets = batch_targets.to(device)

                    optimizer.zero_grad()
                    output = model(batch_inputs, structure_ids)  # (B, N, C)
                    B, N, C = output.shape
                    loss = criterion(
                        output.view(B * N, C),
                        batch_targets.view(-1)
                    )
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_train_loss = epoch_loss / len(train_loader)
                train_epoch_losses.append(avg_train_loss)
                print(f"    Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")

                # Validation per structure
                model.eval()
                with torch.no_grad():
                    for struct in all_structures:
                        val_loader = val_loaders[struct]
                        val_loss = 0.0
                        num_batches = 0
                        for batch_inputs, batch_targets, structure_ids in val_loader:
                            batch_inputs = batch_inputs.to(device)
                            batch_targets = batch_targets.to(device)

                            output = model(batch_inputs, structure_ids)
                            B, N, C = output.shape
                            loss = criterion(
                                output.view(B * N, C),
                                batch_targets.view(-1)
                            )

                            val_loss += loss.item()
                            num_batches += 1

                        avg_val_loss = (
                            val_loss / num_batches if num_batches > 0 else float('nan')
                        )
                        val_epoch_losses[struct].append(avg_val_loss)
                        # print(f"  Val Loss [{struct}]: {avg_val_loss:.4f}")

            for struct in all_structures:
                print(f"Trial 1, Struct {struct}: {val_epoch_losses[struct]}")

            # # Dump edge features
            # mix_tag = "-".join(str(f) for f in frac_tuple)
            # model.dump_edge_features(
            #     out_dir="/home/amgoel/Approx_equiv/SyntheticTaskSeq/logs",
            #     tag=f"multitask_equiv_weakened_mix={mix_tag}__trial=1",
            #     structures=all_structures,
            #     include_all_pairs=True
            # )

            # Final test losses
            model.eval()
            with torch.no_grad():
                for struct in all_structures:
                    test_loader = DataLoader(
                        list(test_datasets[struct]),
                        batch_size=64,
                        shuffle=False,
                        collate_fn=collate_fn
                    )
                    total_loss = 0.0
                    num_batches = 0
                    for batch_inputs, batch_targets, structure_ids in test_loader:
                        batch_inputs = batch_inputs.to(device)
                        batch_targets = batch_targets.to(device)
                        output = model(batch_inputs, structure_ids)
                        B, N, C = output.shape
                        loss = criterion(
                            output.view(B * N, C),
                            batch_targets.view(-1)
                        )
                        total_loss += loss.item()
                        num_batches += 1

                    avg_loss = (
                        total_loss / num_batches if num_batches > 0 else float('nan')
                    )
                    trial_losses[struct].append(avg_loss)

        # Average test loss across trials (here trials=1)
        avg_losses = {
            struct: sum(trial_losses[struct]) / len(trial_losses[struct])
            for struct in all_structures
        }
        results[tuple(frac_tuple)] = avg_losses

    print("\n==== Final Average Test Losses Across Trials ====")
    for setting, loss_dict in results.items():
        frac_tuple = setting
        mix_str = " + ".join(
            f"{frac} {struct}"
            for frac, struct in zip(frac_tuple, all_structures)
        )
        print(f"{mix_str}:")
        for struct in all_structures:
            print(f"  {struct}: {loss_dict[struct]:.4f}")



def run_pretrain_finetune_experiment_equiv():
    # Transfer learning with pretrained model
    import numpy as np

    graph_configs, train_datasets, val_datasets, test_datasets = create_datasets()
    all_structures = list(train_datasets.keys())
    print("Structures:", all_structures)
    
    # Pick finetune target
    finetune_task = all_structures[-1]  
    pretrain_tasks = [s for s in all_structures if s != finetune_task]
    print(f"Pretraining on: {pretrain_tasks} | Fine-tuning on: {finetune_task}")

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

    # Loaders
    def collate_fn(batch):
        xs = torch.stack([item[0] for item in batch])
        ys = torch.stack([item[1] for item in batch])
        struct_ids = [item[2] for item in batch]
        return xs, ys, struct_ids

    pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(finetune_val, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(finetune_test, batch_size=64, shuffle=False, collate_fn=collate_fn)

    num_pretrain_epochs = 15
    num_finetune_epochs = 40

    results = {
        "pretrain+finetune": [],
        "finetune_only": []
    }

    for experiment_type in ["pretrain+finetune", "finetune_only"]:
        print(f"\n====== Starting Experiment: {experiment_type} ======")

        for trial in range(1):
            print(f"\n--- Trial {trial+1} ---")

            model = MultiGraphGATv2Model_equiv(
                graph_configs, hid_dim=128, num_layers=4, p_dropout=0.1, vocab_size=82
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()

            # === Pretraining Phase ===
            if experiment_type == "pretrain+finetune":
                print("  Pretraining...")
                for epoch in range(num_pretrain_epochs):
                    model.train()
                    total_loss = 0.0
                    for x_batch, y_batch, struct_ids in pretrain_loader:
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                        optimizer.zero_grad()
                        override_struct_ids = [finetune_task] * x_batch.size(0)
                        #preds = model(x_batch, override_struct_ids)
                        preds = model(x_batch, struct_ids)  # (B, n_nodes, out_dim)
                        B, N, C = preds.shape
                        loss = criterion(preds.view(B*N, C), y_batch.view(-1))
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()

                    avg_loss = total_loss / len(pretrain_loader)
                    if (epoch+1) % 5 == 0:
                        print(f"    Pretrain Epoch {epoch+1}: Loss = {avg_loss:.4f}")
                dump_path = f"edge_feats__{finetune_task}__pretrain__weirdest__trial={trial+1}.npz"
                model.dump_edge_features(finetune_task, dump_path)

            # === Finetuning Phase ===
            print("  Fine-tuning...")
            finetune_loader = DataLoader(finetune_train, batch_size=64, shuffle=True, collate_fn=collate_fn)
            if experiment_type == "pretrain+finetune":
                gnn_params = []
                edge_embedder_params = []
                token_embedder_params = []
                
                for name, param in model.named_parameters():
                    if "edge_embedders" in name:
                        edge_embedder_params.append(param)
                    elif "token_embedders" in name:
                        token_embedder_params.append(param)
                    else:
                        gnn_params.append(param)
                
                optimizer = torch.optim.Adam([
                    {'params': gnn_params, 'lr': 0.02},            # GNN body
                    {'params': edge_embedder_params, 'lr': 0.02},  # Edge embedders
                    {'params': token_embedder_params, 'lr': 0.02}   # Token embedders
                ])

            for epoch in range(num_finetune_epochs):
                model.train()
                total_loss = 0.0
                for x_batch, y_batch, struct_ids in finetune_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                    optimizer.zero_grad()
                    override_struct_ids = [finetune_task] * x_batch.size(0)
                    preds = model(x_batch, override_struct_ids)
                    
                    # preds = model(x_batch, struct_ids)  # (B, n_nodes, out_dim)
                    B, N, C = preds.shape
                    loss = criterion(preds.view(B*N, C), y_batch.view(-1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_train_loss = total_loss / len(finetune_loader)
                print(f"Trial {trial+1}, Epoch {epoch+1}: Finetune Train Loss = {avg_train_loss:.4f}")
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    num_batches = 0
                    for x_batch, y_batch, struct_ids in val_loader:
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                        preds = model(x_batch, struct_ids)
                        B, N, C = preds.shape
                        loss = criterion(preds.view(B*N, C), y_batch.view(-1))
                        val_loss += loss.item()
                        num_batches += 1
                    avg_val_loss = val_loss / num_batches if num_batches > 0 else float('nan')

                print(f"Trial {trial+1}, Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}")
            
            # dump_path = f"edge_feats__{finetune_task}__{experiment_type}__weirdest__trial={trial+1}.npz"
            # model.dump_edge_features(finetune_task, dump_path)
            # === Testing Phase ===
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                num_batches = 0
                for x_batch, y_batch, struct_ids in test_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    preds = model(x_batch, struct_ids)  # (B, n_nodes, out_dim)
                    B, N, C = preds.shape
                    loss = criterion(preds.view(B*N, C), y_batch.view(-1))
                    test_loss += loss.item()
                    num_batches += 1
                avg_test_loss = test_loss / num_batches
                print(f"  Test Loss: {avg_test_loss:.4f}")

            results[experiment_type].append(avg_test_loss)

    # === Final Results ===
    print("\n====== Final Summary ======")
    for exp_type in results:
        avg_loss = np.mean(results[exp_type])
        std_loss = np.std(results[exp_type])
        print(f"{exp_type}: mean={avg_loss:.4f} std={std_loss:.4f}")





if __name__ == "__main__":
   run_pretrain_finetune_experiment_equiv()
