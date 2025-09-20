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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_graph_structure(dataset_name, seq_length):
    """
    Creates adjacency matrices and orbit mappings for different dataset structures.
    """
    # identity = [i for i in range(seq_length)]    
    # perms=[Permutation(identity)]
    # return perms

    

    if dataset_name == "palindrome":
        # Mirror adjacency (Characters linked to their mirrored counterparts)
        perms = [Permutation([i for i in range(seq_length)]), Permutation([seq_length - i - 1 for i in range(seq_length)])]
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
    
        # # Generate both (i j k) and (i k j) for i < j < k
        # for i in range(seq_length - 2):
        #     for j in range(i + 1, seq_length - 1):
        #         for k in range(j + 1, seq_length):
        #             # 3-cycle: (i j k)
        #             perm1 = identity.copy()
        #             perm1[i], perm1[j], perm1[k] = identity[j], identity[k], identity[i]
        #             perms.append(Permutation(perm1))
    
        #             # 3-cycle: (i k j)
        #             perm2 = identity.copy()
        #             perm2[i], perm2[j], perm2[k] = identity[k], identity[i], identity[j]
        #             perms.append(Permutation(perm2))
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

    structures = ['palindrome','intersect','longestpal','detectcapital']
    #structures = ['intersect','longestpal','detectcapital']
    #structures = ['longestpal','detectcapital', "vandermonde"]
    # structures = ['vandermonde']
    

    for structure_id in structures:
        print(structure_id)
        if structure_id == "palindrome":
            seq_length = 8
            dataset = IsPalindromeDataset(num_samples=4000, seq_length=seq_length, palindrome_length=4,equivariant=False)
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
            seq_length = 8
            dataset = IntersectDataset(num_samples=4000, seq_length=seq_length, vocab_size=9,equivariant=False, thresh = 4)
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
            seq_length = 8
            dataset = MaxCyclicSumDataset(num_samples=4000, seq_length=seq_length, cyc_length = 4, vocab_size=9, inv = True)
            graph_configs[structure_id] = {
            "n_nodes": seq_length,
            "perms": generate_graph_structure("cyclicsum", seq_length),
            "coords_dim": (1,1),
            "adj": None,
            "orbits": None,
            "sparse": False,
            "out_dim": 0
        }
        elif structure_id == "longestpal":
            seq_length = 8
            dataset = LongestPalindromeDataset(num_samples=4000, seq_length=seq_length, vocab_size=9, thresh = 5)
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
            seq_length = 8
            dataset = DetectCapitalDataset(num_samples=4000, word_length=seq_length)
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
            seq_length = 8
            dataset = Vandermonde(num_samples=10000, seq_length=seq_length, vocab_size = 10)
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
    """Multitask learning on invariant datasets with CE loss"""
    import matplotlib.pyplot as plt
    import numpy as np

    graph_configs, train_datasets, val_datasets, test_datasets = create_datasets()
    all_structures = list(train_datasets.keys())
    assert len(all_structures) == 4, "Expected exactly four structures."

    base_size = min(len(train_datasets[s]) for s in all_structures)


    settings=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

 
    #     [0.2,0.2]]

    def collate_fn(batch):
        xs = torch.stack([item[0] for item in batch])
        ys = torch.tensor([item[1] for item in batch])
        struct_ids = [item[2] for item in batch]
        return xs, ys, struct_ids

    def compute_loss_per_sample(preds, targets, struct_ids, graph_configs, criterion):
        losses = []
        for i in range(len(struct_ids)):
            struct = struct_ids[i]
            out_dim = graph_configs[struct]["out_dim"]
            logits = preds[i, :out_dim].unsqueeze(0)         # (1, out_dim)
            label = torch.tensor([targets[i]], device=preds.device)  # (1,)
            loss_i = criterion(logits, label)
            losses.append(loss_i)
        return torch.stack(losses).mean()

    for setting in settings:
        print(f"\nRunning setting: {setting}")
        trial_losses = {s: [] for s in all_structures}

        for trial in range(3):
            subsets = []
            for frac, struct in zip(setting, all_structures):
                n = int(base_size * frac)
                if n > 0:
                    subsets += random.sample(list(train_datasets[struct]), n)
            random.shuffle(subsets)

            train_loader = DataLoader(subsets, batch_size=64, shuffle=True, collate_fn=collate_fn)
            val_loaders = {
                s: DataLoader(val_datasets[s], batch_size=64, shuffle=False, collate_fn=collate_fn)
                for s in all_structures
            }

            model = MultiGraphGATv2Model_inv(
                graph_configs, hid_dim=32, num_layers=2, p_dropout=0.1, vocab_size=53
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
            criterion = torch.nn.CrossEntropyLoss()
            num_epochs = 80

            val_epoch_losses = {s: [] for s in all_structures}

            for epoch in range(num_epochs):
                model.train()
                total_loss = 0.0
                for x_batch, y_batch, struct_ids in train_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    optimizer.zero_grad()
                    preds = model(x_batch, struct_ids)
                    loss = compute_loss_per_sample(preds, y_batch, struct_ids, graph_configs, criterion)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_train_loss = total_loss / len(train_loader)
                print(f"Trial {trial+1}, Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

                # Validation per structure
                model.eval()
                with torch.no_grad():
                    for struct in all_structures:
                        val_loader = val_loaders[struct]
                        val_loss = 0.0
                        num_batches = 0
                        for x_batch, y_batch, struct_ids in val_loader:
                            x_batch = x_batch.to(device)
                            y_batch = y_batch.to(device)

                            preds = model(x_batch, struct_ids)
                            loss = compute_loss_per_sample(preds, y_batch, struct_ids, graph_configs, criterion)

                            val_loss += loss.item()
                            num_batches += 1

                        avg_val_loss = val_loss / num_batches if num_batches > 0 else float('nan')
                        val_epoch_losses[struct].append(avg_val_loss)

            # Print validation loss curves for this trial
            print(f"\nValidation losses for Trial {trial+1}:")
            for struct in all_structures:
                print(f"  {struct}: {val_epoch_losses[struct]}")

            # Final test loss per structure
            model.eval()
            with torch.no_grad():
                for s in all_structures:
                    test_loader = DataLoader(test_datasets[s], batch_size=64, shuffle=False, collate_fn=collate_fn)
                    test_loss = 0.0
                    num_batches = 0
                    for x_batch, y_batch, struct_ids in test_loader:
                        x_batch = x_batch.to(device)
                        y_batch = y_batch.to(device)
                        preds = model(x_batch, struct_ids)
                        loss = compute_loss_per_sample(preds, y_batch, struct_ids, graph_configs, criterion)
                        test_loss += loss.item()
                        num_batches += 1
                    avg_test_loss = test_loss / num_batches
                    trial_losses[s].append(avg_test_loss)

        # Final result summary
        print("  === Final Avg Test Losses ===")
        for s in all_structures:
            print(f"    {s}: {np.mean(trial_losses[s]):.4f}")


def run_pretrain_finetune_experiment():
    """Create pretrained model from datasets, finetune on last dataset using standard invariance datasets with cross-entropy loss"""
    import numpy as np

    graph_configs, train_datasets, val_datasets, test_datasets = create_datasets()
    all_structures = list(train_datasets.keys())
    # assert len(all_structures) == 3, "Expected exactly four structures."

    def collate_fn(batch):
        xs = torch.stack([item[0] for item in batch])
        ys = torch.tensor([item[1] for item in batch])
        struct_ids = [item[2] for item in batch]
        return xs, ys, struct_ids

    def compute_loss_per_sample(preds, targets, struct_ids, graph_configs, criterion):
        losses = []
        for i in range(len(struct_ids)):
            struct = struct_ids[i]
            out_dim = graph_configs[struct]["out_dim"]
            logits = preds[i, :out_dim].unsqueeze(0)
            label = torch.tensor([targets[i]], device=preds.device)
            loss_i = criterion(logits, label)
            losses.append(loss_i)
        return torch.stack(losses).mean()

    # Choose fine-tune task
    finetune_task = "intersect"
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

    finetune_train_size = max(1, int(0.1 * len(finetune_train_full)))
    finetune_train = random.sample(finetune_train_full, finetune_train_size)

    # Loaders
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(finetune_val, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(finetune_test, batch_size=64, shuffle=False, collate_fn=collate_fn)

    num_pretrain_epochs = 20
    num_finetune_epochs = 40
    print(f"Pretrain Epochs: {num_pretrain_epochs}")
    print(f"Finetune Epochs: {num_finetune_epochs}")

    # Record results
    results = {
        "pretrain+finetune": [],
        "finetune_only": []
    }

    for experiment_type in ["pretrain+finetune", "finetune_only"]:
        print(f"\n====== Starting Experiment: {experiment_type} ======")

        for trial in range(3):
            print(f"\n--- Trial {trial+1} ---")

            model = MultiGraphGATv2Model_inv(
                graph_configs, hid_dim=128, num_layers=4, p_dropout=0.1, vocab_size=53
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()

            # === Pretraining Phase ===
            if experiment_type == "pretrain+finetune":
                print("  Pretraining...")
                for epoch in range(num_pretrain_epochs):
                    model.train()
                    total_loss = 0.0
                    for x_batch, y_batch, struct_ids in DataLoader(pretrain_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn):
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                        optimizer.zero_grad()
                        preds = model(x_batch, struct_ids)
                        loss = compute_loss_per_sample(preds, y_batch, struct_ids, graph_configs, criterion)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    avg_loss = total_loss / len(pretrain_dataset)
                    if (epoch+1) % 5 == 0:
                        print(f"    Pretrain Epoch {epoch+1}: Loss = {avg_loss:.4f}")

            # === Fine-tuning Phase ===
            print("  Fine-tuning...")
            gnn_params = []
            edge_embedder_params = []
            
            for name, param in model.named_parameters():
                if "edge_embedders" in name:
                    edge_embedder_params.append(param)
                else:
                    gnn_params.append(param)
            
            # Two groups of parameters with different learning rates
            optimizer = torch.optim.Adam([
                {'params': gnn_params, 'lr': 0.001},       # smaller LR for GNN
                {'params': edge_embedder_params, 'lr': 0.02}  # larger LR for edge embedders
            ])
            finetune_loader = DataLoader(finetune_train, batch_size=64, shuffle=True, collate_fn=collate_fn)
            val_losses = []

            for epoch in range(num_finetune_epochs):
                model.train()
                total_loss = 0.0
                for x_batch, y_batch, struct_ids in finetune_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    preds = model(x_batch, struct_ids)
                    loss = compute_loss_per_sample(preds, y_batch, struct_ids, graph_configs, criterion)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                avg_train_loss = total_loss / len(finetune_loader)
                print(f"Trial {trial+1}, Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    num_batches = 0
                    for x_batch, y_batch, struct_ids in val_loader:
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                        preds = model(x_batch, struct_ids)
                        loss = compute_loss_per_sample(preds, y_batch, struct_ids, graph_configs, criterion)
                        val_loss += loss.item()
                        num_batches += 1
                    avg_val_loss = val_loss / num_batches
                    val_losses.append(avg_val_loss)
                    print(f"    Finetune Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}")

            # === Test Phase ===
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                num_batches = 0
                for x_batch, y_batch, struct_ids in test_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    preds = model(x_batch, struct_ids)
                    loss = compute_loss_per_sample(preds, y_batch, struct_ids, graph_configs, criterion)
                    test_loss += loss.item()
                    num_batches += 1
                avg_test_loss = test_loss / num_batches
                print(f"  Test Loss: {avg_test_loss:.4f}")

            results[experiment_type].append(avg_test_loss)

    # === Final results ===
    print("\n====== Final Summary ======")
    for exp_type in results:
        avg_loss = np.mean(results[exp_type])
        std_loss = np.std(results[exp_type])
        print(f"{exp_type}: mean={avg_loss:.4f} std={std_loss:.4f}")
        
        
        

def run_vandermonde_mlp():
    num_trials = 3
    num_epochs = 600
    batch_size = 64
    seq_length = 5
    hidden_dim = 64

    # Use vocab_size=seq_length to make inputs pure permutations (optional but common)
    full_dataset = Vandermonde(num_samples=6000, seq_length=seq_length, vocab_size=seq_length)

    # Print label distribution
    label_counts = Counter([label.item() for _, label in full_dataset])
    print("Label distribution:", label_counts)

    # Split dataset
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

    def collate_fn(batch):
        xs = torch.stack([item[0] for item in batch]).float()      # (B, L)
        xs = xs.unsqueeze(-1)                                      # (B, L, 1)
        ys = torch.tensor([item[1] for item in batch], dtype=torch.long)  # (B,)
        return xs, ys

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    trial_test_losses = []

    for trial in range(num_trials):
        print(f"\n=== Trial {trial+1} ===")
        model = PermutationMLP(seq_length=seq_length, hidden_dim=hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(x_batch)               # (B, 2)
                loss = criterion(logits, y_batch)
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
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)
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
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))

    print("\n====== Final Summary ======")
    print(f"Mean Test Loss: {np.mean(trial_test_losses):.4f}")
    print(f"Std Dev:        {np.std(trial_test_losses):.4f}")






if __name__ == "__main__":
   run_experiments_inv()
