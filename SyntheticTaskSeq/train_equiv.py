import torch
import pickle
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from models.Multi_GAT import MultiGraphGATv2Model_inv,MultiGraphGATv2Model_equiv
from sympy import *
from sympy.combinatorics import Permutation, PermutationGroup
from random import randrange
from data import IsBalancedParenthesisDataset, IsPalindromeDataset, IntersectDataset, MaxCyclicSumDataset  # Import dataset classes
from torch.utils.data import ConcatDataset, DataLoader
from collections import deque

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
        oppositeidentity = [seq_length-1-i for i in range(seq_length)]
        perms = [Permutation(identity), Permutation(oppositeidentity)]
        for i in range(seq_length):
            my_list = deque(identity)
            my_list.rotate(i)  # rotate right by 1
            perms.append(Permutation(list(my_list)))

        
            

    elif dataset_name == "intersect":
        # Bipartite graph: First half linked to second half
        mid = seq_length//2
        identity = [i for i in range(seq_length)]    
        perms=[Permutation(identity)]
        for i in range(1, seq_length):
            if i == mid:
                continue
            perm = identity.copy()  # or use deepcopy if needed
            perm[i], perm[i-1] = perm[i-1], perm[i]
            perms.append(Permutation(perm))
        perms.append(Permutation([(i + mid) % seq_length for i in range(seq_length)])) 
        

    

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
    # structures = ["palindrome","intersect"]
    # structures = ["palindrome"]
    # structures = ["intersect"]
    structures = ["cyclicsum", "intersect", "palindrome"]
    

    for structure_id in structures:
        print(structure_id)
        if structure_id == "palindrome":
            seq_length = 10
            dataset = IsPalindromeDataset(num_samples=10000, seq_length=seq_length, palindrome_length=4,equivariant=True)
            graph_configs[structure_id] = {
            "n_nodes": 10,
            "perms": generate_graph_structure("palindrome", seq_length),
            "coords_dim": (1,1),
            "adj": None,
            "orbits": None,
            "sparse": False,
            "out_dim": 2
        }
        elif structure_id == "intersect":
            print(f"Vocab size: 5")
            seq_length = 10
            dataset = IntersectDataset(num_samples=10000, seq_length=seq_length, vocab_size=5,equivariant=True)
            graph_configs[structure_id] = {
            "n_nodes": 10,
            "perms": generate_graph_structure("intersect", seq_length),
            "coords_dim": (1,1),
            "adj": None,
            "orbits": None,
            "sparse": False,
            "out_dim": 4
        }
        elif structure_id == "cyclicsum":
            seq_length = 10
            dataset = MaxCyclicSumDataset(num_samples=10000, seq_length=seq_length, cyc_length = 3, vocab_size=5)
            graph_configs[structure_id] = {
            "n_nodes": 10,
            "perms": generate_graph_structure("cyclicsum", seq_length),
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

def train_equivariant():
    graph_configs, train_datasets, val_datasets, test_datasets = create_datasets()

    model = MultiGraphGATv2Model_equiv(graph_configs, hid_dim=128, num_layers=4, p_dropout=0.1, vocab_size=20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    structures = list(train_datasets.keys())

    train_loaders = {
        struct: DataLoader(train_datasets[struct], batch_size=64, shuffle=True)
        for struct in structures
    }

    test_loaders = {
        struct: DataLoader(test_datasets[struct], batch_size=64, shuffle=False)
        for struct in structures
    }

    from itertools import cycle
    train_iters = {k: cycle(v) for k, v in train_loaders.items()}

    num_epochs = 40
    steps_per_epoch = 100

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            struct = random.choice(structures)
            batch_inputs, batch_targets, structure_ids = next(train_iters[struct])

            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            output = model(batch_inputs, structure_ids[0])  # output shape = (batch, n_nodes, out_dim)

            # Reshape output and targets for per-node CE loss
            B, N, C = output.shape
            loss = criterion(output.view(B * N, C), batch_targets.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    print("Training complete!\nPer-Structure Test Loss:")

    model.eval()
    with torch.no_grad():
        for struct in structures:
            test_loader = test_loaders[struct]
            total_loss = 0.0
            num_batches = 0

            for batch_inputs, batch_targets, structure_ids in test_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)

                output = model(batch_inputs, structure_ids[0])
                B, N, C = output.shape
                loss = criterion(output.view(B * N, C), batch_targets.view(-1))

                total_loss += loss.item()
                num_batches += 1

            avg_test_loss = total_loss / num_batches if num_batches > 0 else float('nan')
            print(f"  {struct}: Avg CE Loss = {avg_test_loss:.4f}")


def train_equiv():
    graph_configs, train_datasets, val_datasets, test_datasets = create_datasets()

    model = MultiGraphGATv2Model_equiv(graph_configs, hid_dim=128, num_layers=4, p_dropout=0.1, vocab_size=20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Flatten all train datasets into one
    all_train_data = []
    for struct in train_datasets:
        all_train_data.extend(train_datasets[struct])  # each item is (input, label, structure_id)

    random.shuffle(all_train_data)  # ensure random mix

    def collate_fn(batch):
        xs = torch.stack([item[0] for item in batch])  # (B, num_nodes, 1)
        ys = torch.stack([item[1] for item in batch])  # (B, num_nodes)
        struct_ids = [item[2] for item in batch]       # List[str] of length B
        return xs, ys, struct_ids

    train_loader = DataLoader(all_train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)

    # Flatten and prepare test datasets per structure
    test_loaders = {}
    for struct in test_datasets:
        test_data = list(test_datasets[struct])
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate_fn)
        test_loaders[struct] = test_loader

    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_inputs, batch_targets, structure_ids in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            output = model(batch_inputs, structure_ids)  # output shape = (batch, n_nodes, out_dim)
            B, N, C = output.shape
            loss = criterion(output.view(B * N, C), batch_targets.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    print("Training complete!\nPer-Structure Test Loss:")

    model.eval()
    with torch.no_grad():
        for struct in test_loaders:
            test_loader = test_loaders[struct]
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

            avg_test_loss = total_loss / num_batches if num_batches > 0 else float('nan')
            print(f"  {struct}: Avg CE Loss = {avg_test_loss:.4f}")




def run_experiments():
    import copy
    import matplotlib.pyplot as plt
    import numpy as np

    graph_configs, train_datasets, val_datasets, test_datasets = create_datasets()
    all_structures = list(train_datasets.keys())
    assert len(all_structures) == 3, "Currently supports three datasets only."

    s1, s2, s3 = all_structures
    base_size = len(train_datasets[s1]) // 4  # Equal-sized base

    settings = [
        (0.333, 0.333, 0.333),   
        (1.0, 0.0, 0.0),   
        (0.0, 1.0, 0.0),   
        (0.0, 0.0, 1.0),
        (0.6, 0.2, 0.2), 
        (0.2, 0.6, 0.2),  
        (0.2, 0.2, 0.6),
        (0.6,0,0),
        (0,0.6,0),
        (0,0,0.6)
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

        for trial in range(3):
            print(f"  Trial {trial + 1}/3")

            # Sample training and validation sets
            n_total = int(base_size)
            n1 = int(n_total * frac1)
            n2 = int(n_total * frac2)
            n3 = int(n_total * frac3)

            subset1 = random.sample(list(train_datasets[s1]), n1) if n1 > 0 else []
            subset2 = random.sample(list(train_datasets[s2]), n2) if n2 > 0 else []
            subset3 = random.sample(list(train_datasets[s3]), n3) if n3 > 0 else []
            combined_train = subset1 + subset2 + subset3
            random.shuffle(combined_train)
            train_loader = DataLoader(combined_train, batch_size=64, shuffle=True, collate_fn=collate_fn)

            val_loaders = {
                    struct: DataLoader(list(val_datasets[struct]), batch_size=64, shuffle=False, collate_fn=collate_fn)
                    for struct in val_datasets
                }

            model = MultiGraphGATv2Model_equiv(
                graph_configs, hid_dim=128, num_layers=4, p_dropout=0.1, vocab_size=20
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
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

        # # Plot average train/val loss
        # avg_train_loss_per_epoch = np.mean(train_losses_over_trials, axis=0)
        # avg_val_loss_per_epoch = np.mean(val_losses_over_trials, axis=0)

        # plt.figure()
        # plt.plot(avg_train_loss_per_epoch, label="Train Loss")
        # plt.plot(avg_val_loss_per_epoch, label="Validation Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title(f"Setting: {frac1} {s1} + {frac2} {s2} + {frac3} {s3}")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

    print("\n==== Final Average Test Losses Across Trials ====")
    for setting, loss_dict in results.items():
        frac1, frac2, frac3 = setting
        print(f"{frac1}  {s1} + {frac2}  {s2} + {frac3} {s3}:")
        print(f"  {s1}: {loss_dict[s1]:.4f}")
        print(f"  {s2}: {loss_dict[s2]:.4f}")
        print(f"  {s3}: {loss_dict[s3]:.4f}")




if __name__ == "__main__":
    run_experiments()
