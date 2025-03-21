


import argparse
import os
import time
import pickle
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model.pytorch.GATV2_cell import GATv2Seq2SeqModel  # Import your model
from lib.utils import load_graph_data  # Utility functions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[loss != loss] = 0  # Replace NaNs with 0
    return loss.mean()


def load_data(train_path, val_path, test_path):
    """
    Load METR-LA dataset from .npz files.
    """
    train_npz = np.load(train_path)
    val_npz = np.load(val_path)
    test_npz = np.load(test_path)

    train_x, train_y = train_npz["x"].astype(np.float32), train_npz["y"].astype(np.float32)
    val_x, val_y = val_npz["x"].astype(np.float32), val_npz["y"].astype(np.float32)
    test_x, test_y = test_npz["x"].astype(np.float32), test_npz["y"].astype(np.float32)

    # Convert to PyTorch tensors
    train_inputs = torch.tensor(train_x).permute(0, 2, 1, 3).reshape(train_x.shape[0], 207, -1)
    train_targets = torch.tensor(train_y, dtype=torch.float)

    val_inputs = torch.tensor(val_x).permute(0, 2, 1, 3).reshape(val_x.shape[0], 207, -1)
    val_targets = torch.tensor(val_y, dtype=torch.float)

    test_inputs = torch.tensor(test_x).permute(0, 2, 1, 3).reshape(test_x.shape[0], 207, -1)
    test_targets = torch.tensor(test_y, dtype=torch.float)

    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)

    return train_dataset, val_dataset, test_dataset


def train(args):
    # Load graph structure
    with open(args.orbits, "rb") as file:
        orbits = pickle.load(file)
    orbits = orbits[0]

    with open(args.config_filename) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    graph_pkl_filename = config["data"]["graph_pkl_filename"]
    sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
    adj = adj_mx[2]  # Shape (207, 207)

    # Load dataset
    train_dataset, val_dataset, test_dataset = load_data(
        config["data"]["train_path"],
        config["data"]["val_path"],
        config["data"]["test_path"],
    )

    # Initialize model
    model = GATv2Seq2SeqModel(
        n_nodes=args.n_nodes,
        hid_dim=args.hid_dim,
        orbits=orbits,
        adj=adj,
        num_layers=args.num_layers,
        sparse=args.sparse
    ).to(device)

    print("Total parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    # Move edge information to GPU
    edge_index = model.encoder.gat_layers[0].gat.edge_index.to(device)
    edge_categories = model.encoder.gat_layers[0].gat.edge_categories.to(device)

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    criterion = masked_mae_loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = args.epochs
    best_val_loss = float("inf")

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_inputs, batch_targets in train_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            optimizer.zero_grad()

            # Initialize hidden state properly (batch_size, num_nodes, hid_dim)
            hidden_state = torch.zeros(1, batch_inputs.shape[0], args.n_nodes, args.hid_dim, device=device)

            # Step 1: Encode input sequence to get hidden state
            encoder_output, hidden_state = model.encoder(batch_inputs, hidden_state)

            # Step 2: Decode using autoregressive method
            decoder_input = batch_inputs[:, -1, :, :]  # Last input frame
            outputs = []

            for t in range(batch_targets.shape[1]):
                decoder_output, hidden_state = model.decoder(
                    decoder_input, hidden_state
                )
                outputs.append(decoder_output)

                # Teacher forcing
                decoder_input = batch_targets[:, t, :, :] if torch.rand(1).item() < args.teacher_forcing_ratio else decoder_output

            outputs = torch.stack(outputs, dim=1)  # (batch_size, horizon, num_nodes, out_dim)

            # Compute loss
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.6f} | Time: {epoch_time:.2f}s")

        # Save best model
        if avg_train_loss < best_val_loss:
            best_val_loss = avg_train_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_gatv2_model.pth")
            print("Saved best model at epoch", epoch + 1)

    print("Training complete. Running final test...")

    # Testing Loop
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            # Initialize hidden state
            hidden_state = torch.zeros(batch_inputs.shape[0], args.n_nodes, args.hid_dim, device=device)

            encoder_output, hidden_state = model.encoder(batch_inputs, hidden_state)

            decoder_input = batch_inputs[:, -1, :, :]
            outputs = []

            for t in range(batch_targets.shape[1]):
                decoder_output, hidden_state = model.decoder(
                    decoder_input, hidden_state
                )
                outputs.append(decoder_output)
                decoder_input = decoder_output  # Autoregressive inference

            outputs = torch.stack(outputs, dim=1)
            loss_test = criterion(outputs, batch_targets)
            test_loss += loss_test.item()

    avg_test_loss = test_loss / len(test_loader)
    print("Final Test Loss: {:.6f}".format(avg_test_loss))


def main():
    parser = argparse.ArgumentParser(description="Train GATv2 Seq2Seq Model")
    parser.add_argument("--config_filename", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--orbits", type=str, default="orbit_idx.p", help="Path to orbits pickle file")
    parser.add_argument("--n_nodes", type=int, default=207)
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument('--in_features', type=int, default=6, help="Input feature dimension")
    parser.add_argument('--out_features', type=int, default=2, help="Output feature dimension")
    parser.add_argument('--sparse', action='store_true', help="Use sparsified graph")


    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

