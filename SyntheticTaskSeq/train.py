import torch
import pickle
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from models.Multi_GAT import MultiGraphGATv2Model_inv
from sympy import *
from sympy.combinatorics import Permutation, PermutationGroup
from random import randrange
from data import IsBalancedParenthesisDataset, IsPalindromeDataset, IntersectDataset  # Import dataset classes
from torch.utils.data import ConcatDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_graph_structure(dataset_name, seq_length):
    """
    Creates adjacency matrices and orbit mappings for different dataset structures.
    """
    adj_matrix = np.zeros((seq_length, seq_length))

    # if dataset_name == "balance":
    #     # Chain-like adjacency (Each character depends on the previous)
    #     for i in range(seq_length - 1):
    #         adj_matrix[i, i + 1] = 1
    #         adj_matrix[i + 1, i] = 1
    

    if dataset_name == "palindrome":
        # Mirror adjacency (Characters linked to their mirrored counterparts)
        perms = [Permutation([i for i in range(seq_length)]), Permutation([seq_length - i - 1 for i in range(seq_length)])]

    elif dataset_name == "intersect":
        # Bipartite graph: First half linked to second half
        mid = seq_length//2
        identity = [i for i in range(seq_length)]    
        perms=[Permutation(identity)]
        for i in range(1,seq_length):
            if(i == mid):
                continue
            perm = identity
            temp = perm[i]
            perm[i] = perm[i-1]
            perm[i-1] = temp
            perms.append(Permutation(perm))
        perms.append(Permutation([(i + mid) % seq_length for i in range(seq_length)])) 
        

    

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    return perms



def create_datasets():
    """
    Creates datasets for multiple graph structures and returns a combined dataset with labels.
    """
    graph_configs = {}
    dataset_splits = {}
    train_data = {}
    test_data={}

    # Define structure names (or structure IDs)
    structures = ["palindrome", "intersect"]

    for structure_id in structures:
        if structure_id == "palindrome":
            seq_length = 6
            dataset = IsPalindromeDataset(num_samples=10000, seq_length=seq_length, palindrome_length=3)
            graph_configs[structure_id] = {
            "n_nodes": 6,
            "perms": generate_graph_structure("palindrome", seq_length),
            "coords_dim": (1,1),
            "adj": None,
            "orbits": None,
            "sparse": False
        }
        elif structure_id == "intersect":
            seq_length = 6
            dataset = IntersectDataset(num_samples=10000, seq_length=seq_length, vocab_size=4)
            graph_configs[structure_id] = {
            "n_nodes": 6,
            "perms": generate_graph_structure("intersect", seq_length),
            "coords_dim": (1,1),
            "adj": None,
            "orbits": None,
            "sparse": False
        }

        # Add `structure_id` as metadata in dataset
        labeled_dataset = [(seq.unsqueeze(1), int(label), structure_id) for seq, label in dataset]


        # Split into train and test
        train_size = int(0.8 * len(labeled_dataset))
        test_size = len(labeled_dataset) - train_size
        train_dataset, test_dataset = random_split(labeled_dataset, [train_size, test_size])
        train_data[structure_id] = train_dataset
        test_data[structure_id] = test_dataset
 



    return graph_configs, train_data, test_data





def train_model():
    graph_configs, train_datasets, test_datasets = create_datasets()  

    # Initialize model
    model = MultiGraphGATv2Model_inv(graph_configs, hid_dim=128, num_layers=4, p_dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.L1Loss()

    structures = list(train_datasets.keys())

    # Create DataLoaders per structure
    train_loaders = {
        struct: DataLoader(train_datasets[struct], batch_size=64, shuffle=True)
        for struct in structures
    }

    test_loaders = {
        struct: DataLoader(test_datasets[struct], batch_size=64, shuffle=False)
        for struct in structures
    }

    # Wrap each DataLoader in an infinite cycle
    from itertools import cycle
    train_iters = {k: cycle(v) for k, v in train_loaders.items()}

    num_epochs = 40
    steps_per_epoch = 100  # Number of random batches per epoch

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            # Randomly pick one structure
            struct = random.choice(structures)
            batch_inputs, batch_targets, structure_ids = next(train_iters[struct])

            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            output = model(batch_inputs, structure_ids[0])  # Safe: all structure_ids in batch are the same
            loss = criterion(output.squeeze(), batch_targets.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    print("Training complete!")

    model.eval()
    print("Per-Structure Test Loss:")

    with torch.no_grad():
        for struct in structures:
            test_loader = test_loaders[struct]
            total_loss = 0.0
            num_batches = 0

            for batch_inputs, batch_targets, structure_ids in test_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)

                output = model(batch_inputs, structure_ids[0])  # all structure_ids in batch are the same
                loss = criterion(output.squeeze(), batch_targets.float())

                total_loss += loss.item()
                num_batches += 1

            avg_test_loss = total_loss / num_batches if num_batches > 0 else float('nan')
            print(f"  {struct}: Avg L1 Loss = {avg_test_loss:.4f}")


if __name__ == "__main__":
    train_model()



