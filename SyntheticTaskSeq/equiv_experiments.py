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
        # Mirror adjacency (Characters linked to their mirrored counterparts)
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
    

    for structure_id in structures:
        print(structure_id)
        if structure_id == "palindrome":
            seq_length = 10
            dataset = IsPalindromeDataset(num_samples=100, seq_length=seq_length, palindrome_length=4,equivariant=True)
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
            seq_length = 10
            dataset = IntersectDataset(num_samples=100, seq_length=seq_length, vocab_size=13,equivariant=True)
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
            seq_length = 10
            dataset = MaxCyclicSumDataset(num_samples=100, seq_length=seq_length, cyc_length = 4, vocab_size=13)
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
            seq_length = 12
            dataset = SetGameDataset(num_samples=3500, seq_length=seq_length)
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
            seq_length = 12
            dataset = SETIntersect(num_samples=1300, seq_length=seq_length)
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



def run_experiments():
    # Equivariant Multitask learning
    import copy
    import matplotlib.pyplot as plt
    import numpy as np

    graph_configs, train_datasets, val_datasets, test_datasets = create_datasets()
    all_structures = list(train_datasets.keys())
    assert len(all_structures) == 3, "Currently supports three datasets only."

    s1, s2, s3 = all_structures
    base_size = len(train_datasets[s1])  # Equal-sized base

 
    settings = [
        (0, 0.6, 0),   
        (0, 1, 0), 
        (0.6, 0, 0),  
        (1, 0, 0),
        (0.6, 0.6, 0.6),
        (1, 1, 1),   
    ]


    def collate_fn(batch):
        xs = torch.stack([item[0] for item in batch])  
        ys = torch.stack([item[1] for item in batch])  
        struct_ids = [item[2] for item in batch]       
        return xs, ys, struct_ids

    results = {}

    for (frac1, frac2, frac3) in settings:
        print(f"\nRunning setting: {frac1} {s1} + {frac2}  {s2} + {frac3} {s3}")
        trial_losses = {s1: [], s2: [], s3: []}
        train_losses_over_trials = []
        val_losses_over_trials = []
        n_total = int(base_size)
        n1 = int(n_total * frac1)
        n2 = int(n_total * frac2)
        n3 = int(n_total * frac3)
        subset1 = random.sample(list(train_datasets[s1]), n1) if n1 > 0 else []
        subset2 = random.sample(list(train_datasets[s2]), n2) if n2 > 0 else []
        subset3 = random.sample(list(train_datasets[s3]), n3) if n3 > 0 else []
        combined_train = subset1 + subset2 + subset3
        for trial in range(3):
            print(f"  Trial {trial + 1}/3")





            
            random.shuffle(combined_train)
            train_loader = DataLoader(combined_train, batch_size=64, shuffle=True, collate_fn=collate_fn)

            val_loaders = {
                    struct: DataLoader(list(val_datasets[struct]), batch_size=64, shuffle=False, collate_fn=collate_fn)
                    for struct in val_datasets
                }

            model = MultiGraphGATv2Model_equiv(
                graph_configs, hid_dim=128, num_layers=4, p_dropout=0.1, vocab_size=81
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
                {'params': gnn_params, 'lr': 0.01},            # GNN body
                {'params': edge_embedder_params, 'lr': 0.03},  # Edge embedders
                {'params': token_embedder_params, 'lr': 0.1}   # Token embedders
            ])
            criterion = torch.nn.CrossEntropyLoss()
            num_epochs = 40

            train_epoch_losses = []
            val_epoch_losses = {}
            for struct in all_structures:
                val_epoch_losses[struct] = []
            

            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0.0

                for batch_inputs, batch_targets, structure_ids in train_loader:
                    batch_inputs = batch_inputs.to(device)
                    batch_targets = batch_targets.to(device)
                    optimizer.zero_grad()
                    output = model(batch_inputs, structure_ids)
                    B, N, C = output.shape
                    loss = criterion(output.view(B * N, C), batch_targets.view(-1))
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
                            loss = criterion(output.view(B * N, C), batch_targets.view(-1))
        
                            val_loss += loss.item()
                            num_batches += 1
        
                        avg_val_loss = val_loss / num_batches if num_batches > 0 else float('nan')
                        val_epoch_losses[struct].append(avg_val_loss)
                        #print(f"  Val Loss [{struct}]: {avg_val_loss:.4f}")
            

            for struct in all_structures:
                print(f"Trial {trial+1}, Struct {struct}: {val_epoch_losses[struct]}")


            # Final test losses
            model.eval()
            with torch.no_grad():
                for struct in [s1, s2, s3]:
                    test_loader = DataLoader(list(test_datasets[struct]), batch_size=64, shuffle=False, collate_fn=collate_fn)
                    total_loss = 0.0
                    num_batches = 0
                    for batch_inputs, batch_targets, structure_ids in test_loader:
                        batch_inputs = batch_inputs.to(device)
                        batch_targets = batch_targets.to(device)
                        output = model(batch_inputs, structure_ids)
                        B, N, C = output.shape
                        loss = criterion(output.view(B * N, C), batch_targets.view(-1))
                        total_loss += loss.item()
                        num_batches += 1
                    avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
                    trial_losses[struct].append(avg_loss)

        # Average test loss across trials
        avg_s1 = sum(trial_losses[s1]) / len(trial_losses[s1])
        avg_s2 = sum(trial_losses[s2]) / len(trial_losses[s2])
        avg_s3 = sum(trial_losses[s3]) / len(trial_losses[s3])
        results[(frac1, frac2, frac3)] = {s1: avg_s1, s2: avg_s2, s3: avg_s3}


    print("\n==== Final Average Test Losses Across Trials ====")
    for setting, loss_dict in results.items():
        frac1, frac2, frac3= setting
        print(f"{frac1}  {s1} + {frac2}  {s2} + {frac3} {s3}:")
        print(f"  {s1}: {loss_dict[s1]:.4f}")
        print(f"  {s2}: {loss_dict[s2]:.4f}")
        print(f"  {s3}: {loss_dict[s3]:.4f}")


def run_pretrain_finetune_experiment_equiv():
    # Transfer learning with pretrained model
    import numpy as np

    graph_configs, train_datasets, val_datasets, test_datasets = create_datasets()
    all_structures = list(train_datasets.keys())
    print("Structures:", all_structures)
    
    # Pick finetune target
    finetune_task = "set"  # you can change this
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

    finetune_train_size = max(1, int(0.2 * len(finetune_train_full)))
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

        for trial in range(3):
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
                        preds = model(x_batch, struct_ids)  # (B, n_nodes, out_dim)
                        B, N, C = preds.shape
                        loss = criterion(preds.view(B*N, C), y_batch.view(-1))
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()

                    avg_loss = total_loss / len(pretrain_loader)
                    if (epoch+1) % 5 == 0:
                        print(f"    Pretrain Epoch {epoch+1}: Loss = {avg_loss:.4f}")

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
                    preds = model(x_batch, struct_ids)  # (B, n_nodes, out_dim)
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
   run_experiments()