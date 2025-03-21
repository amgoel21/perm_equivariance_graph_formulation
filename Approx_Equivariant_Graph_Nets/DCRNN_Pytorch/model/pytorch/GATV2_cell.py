import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EdgeEmbedder(nn.Module):
    def __init__(self, num_categories, embedding_dim):
        """
        Initialize the edge embedder with category-specific max norms.
        Args:
            num_categories: Number of unique edge categories.
            embedding_dim: Dimension of the embedding vectors.
        """
        super(EdgeEmbedder, self).__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim, max_norm=None)

    def forward(self, category_indices):
        category_indices = category_indices.to(device) 
        embeddings = self.embedding(category_indices)
        return embeddings



def index(orbits):
    pos = [0]*207
    for a in range(len(orbits)):
        for b in orbits[a]:
            pos[b]=a
    return pos

def sparsify(pos,adj,orbits):
    edge_index=[]
    edge_categories=[[] for i in range(len(orbits)+len(orbits)**2)]
    for i in range(207):
        for j in range(207):
            if adj[i][j]==1:
                if(i==j):
                    edge_index.append((i,i))
                    i_orbit = pos[i]
                    edge_categories[len(orbits)**2+i_orbit].append((i,j))
                else:
                    edge_index.append((i,j))
                    i_orbit = pos[i]
                    j_orbit = pos[j]
                    edge_categories[i_orbit*len(orbits)+j_orbit].append((i,j))
    return len(orbits)+len(orbits)**2, edge_index,edge_categories

def find_categories(pos,orbits):
    n = len(orbits)
    edge_categories=[[] for i in range(n+n**2)]
    for i in range(207):
        for j in range(207):
            if(i==j):
                i_orbit = pos[i]
                edge_categories[len(orbits)**2+i_orbit].append((i,j))
            else:
                i_orbit = pos[i]
                j_orbit = pos[j]
                edge_categories[i_orbit*len(orbits)+j_orbit].append((i,j))
    return edge_categories


def compute_closure(n, adj, orbits, sparse):
    pos = index(orbits)
    if sparse:
        num_categories, edge_index, edge_orbits, = sparsify(pos,adj,orbits)
        edge_to_category = {}
        for cat_idx, orbit in enumerate(edge_orbits):
            for edge in orbit:
                edge_to_category[edge] = cat_idx
        edge_categories = []
        for edge in edge_index:
                edge_categories.append(edge_to_category[edge])
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_categories = torch.tensor(edge_categories)
    else:
        edge_indices = [(i, j) for i in range(n) for j in range(n)]
        edge_orbits = find_categories(pos,orbits)
        num_categories = len(edge_orbits)

        # Create category mapping for each edge
        edge_to_category = {}
        for cat_idx, orbit in enumerate(edge_orbits):
            for edge in orbit:
                edge_to_category[edge] = cat_idx
        edge_index = []
        edge_categories = []
        for i in range(n):
            for j in range(n):
                edge_index.append([i, j])
                edge_categories.append(edge_to_category[(i, j)])
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_categories = torch.tensor(edge_categories)
    return num_categories, edge_index, edge_categories
    

# ======= GATv2 Model (Uses Orbits to Initialize Edge Index) =======
# class GATv2Model(nn.Module):
#     def __init__(self, n_nodes, hid_dim, orbits, adj, coords_dim=(6, 2), num_layers=4, p_dropout=None, sparse=False):
#         super(GATv2Model, self).__init__()
#         self.n = n_nodes  # number of nodes
#         self.hid_dim = hid_dim
#         self.num_categories, edge_index, edge_categories = compute_closure(self.n,adj, orbits, sparse)
#         self.edge_index = torch.tensor(edge_index).t().contiguous().to(device)
#         self.edge_categories = torch.tensor(edge_categories).to(device)
#         self.edge_embedder = EdgeEmbedder(self.num_categories, embedding_dim=hid_dim)
#         self.lin_edge = nn.Linear(hid_dim, hid_dim)
#         # GAT layers
#         self.input_mlp = nn.Sequential(
#             nn.Linear(coords_dim[0], hid_dim),
#             nn.LayerNorm(hid_dim),
#             nn.ReLU(),
#             nn.Linear(hid_dim, hid_dim),
#             nn.LayerNorm(hid_dim)
#         )

#         self.num_heads = 8
#         self.head_dim = hid_dim // self.num_heads

#         conv_layers = []
#         layer_norms = []
#         projection_layers = []

#         for _ in range(num_layers):
#             conv = GATv2Conv(hid_dim, self.head_dim, edge_dim=hid_dim, heads=8, concat=True)
#             conv_layers.append(conv)
#             layer_norms.append(nn.LayerNorm(hid_dim))

#             # Projection layer to match concatenated output to input dimension
#             projection_layers.append(nn.Linear(hid_dim, hid_dim))

#         self.conv_layers = nn.ModuleList(conv_layers)
#         self.layer_norms = nn.ModuleList(layer_norms)
#         self.projection_layers = nn.ModuleList(projection_layers)

#         # Output layer
#         self.output_layer = nn.Linear(hid_dim, coords_dim[1])

#         # Dropout
#         self.dropout = nn.Dropout(p_dropout) if p_dropout is not None else None
#     def forward(self, x):
#         """
#         Forward pass for batched input.
#         Args:
#             x: Tensor of shape (batch_size, n_nodes, coords_dim[0])
#         Returns:
#             Tensor of shape (batch_size, n_nodes, coords_dim[1])
#         """
#         device = x.device
#         batch_size, num_nodes, _ = x.size()

#         # Expand edge indices and edge features for batching
#         edge_index = self.edge_index.to(device)
#         edge_categories = self.edge_categories.to(device)

#         # Create edge features for the batch
#         edge_features = self.edge_embedder(edge_categories).unsqueeze(0).expand(batch_size, -1, -1)

#         # Flatten batch and node dimensions for compatibility with GATConv
#         h = x.view(batch_size * num_nodes, -1)
#         h = self.input_mlp(h)

#         # Create batched edge index for PyG
#         batch_offset = torch.arange(batch_size, device=device).repeat_interleave(edge_index.size(1)) * num_nodes
#         batched_edge_index = edge_index.repeat(1, batch_size) + batch_offset

#         # Process through GATConv layers
#         for conv, ln, proj in zip(self.conv_layers, self.layer_norms, self.projection_layers):
#             h_prev = h
#             h = conv(h, batched_edge_index, edge_features.reshape(-1, edge_features.size(-1)))
#             h = proj(h)  # Project concatenated heads to match input dimension
#             h = ln(h)
#             h = F.relu(h)
#             h = h + h_prev  # Residual connection

#         # Reshape back to (batch_size, n_nodes, hid_dim)
#         h = h.view(batch_size, num_nodes, -1)

#         # Output layer
#         output = self.output_layer(h)

#         # Apply dropout if training
#         if self.dropout is not None and self.training:
#             output = self.dropout(output)

#         return output

class GATv2Model(nn.Module):
    """
    GAT-based model with permutation-aware edge features and concatenated attention heads.
    """
    def __init__(self, n_nodes, hid_dim, orbits, adj, coords_dim=(6, 2), num_layers=4, p_dropout=None, sparse=False):
        super(GATv2Model, self).__init__()
        self.n = n_nodes  # number of nodes
        self.hid_dim = hid_dim
        pos = index(orbits)
        if sparse:
            self.num_categories, edge_index, edge_orbits, = sparsify(pos,adj,orbits)
            self.edge_to_category = {}
            for cat_idx, orbit in enumerate(edge_orbits):
                for edge in orbit:
                    self.edge_to_category[edge] = cat_idx

            # Edge embedder
            self.edge_embedder = EdgeEmbedder(self.num_categories, embedding_dim=hid_dim)
            edge_categories = []
            for edge in edge_index:
                    edge_categories.append(self.edge_to_category[edge])
            self.edge_index = torch.tensor(edge_index).t().contiguous()
            self.edge_categories = torch.tensor(edge_categories)
        else:
            edge_indices = [(i, j) for i in range(self.n) for j in range(self.n)]
            self.edge_orbits = find_categories(pos,orbits)
            self.num_categories = len(self.edge_orbits)

            # Create category mapping for each edge
            self.edge_to_category = {}
            for cat_idx, orbit in enumerate(self.edge_orbits):
                for edge in orbit:
                    self.edge_to_category[edge] = cat_idx

            # Edge embedder
            self.edge_embedder = EdgeEmbedder(self.num_categories, embedding_dim=hid_dim)
            edge_index = []
            edge_categories = []
            for i in range(self.n):
                for j in range(self.n):
                    edge_index.append([i, j])
                    edge_categories.append(self.edge_to_category[(i, j)])
            self.edge_index = torch.tensor(edge_index).t().contiguous()
            self.edge_categories = torch.tensor(edge_categories)

        self.lin_edge = nn.Linear(hid_dim, hid_dim)
        # GAT layers
        self.input_mlp = nn.Sequential(
            nn.Linear(coords_dim[0], hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim)
        )

        self.num_heads = 8
        self.head_dim = hid_dim // self.num_heads

        conv_layers = []
        layer_norms = []
        projection_layers = []

        for _ in range(num_layers):
            conv = GATv2Conv(hid_dim, self.head_dim, edge_dim=hid_dim, heads=8, concat=True)
            conv_layers.append(conv)
            layer_norms.append(nn.LayerNorm(hid_dim))

            # Projection layer to match concatenated output to input dimension
            projection_layers.append(nn.Linear(hid_dim, hid_dim))

        self.conv_layers = nn.ModuleList(conv_layers)
        self.layer_norms = nn.ModuleList(layer_norms)
        self.projection_layers = nn.ModuleList(projection_layers)

        # Output layer
        self.output_layer = nn.Linear(hid_dim, coords_dim[1])

        # Dropout
        self.dropout = nn.Dropout(p_dropout) if p_dropout is not None else None

    def forward(self, x):
        """
        Forward pass for batched input.
        Args:
            x: Tensor of shape (batch_size, n_nodes, coords_dim[0])
        Returns:
            Tensor of shape (batch_size, n_nodes, coords_dim[1])
        """
        device = x.device
        batch_size, num_nodes, _ = x.size()

        # Expand edge indices and edge features for batching
        edge_index = self.edge_index.to(device)
        edge_categories = self.edge_categories.to(device)

        # Create edge features for the batch
        edge_features = self.edge_embedder(edge_categories).unsqueeze(0).expand(batch_size, -1, -1)

        # Flatten batch and node dimensions for compatibility with GATConv
        h = x.view(batch_size * num_nodes, -1)
        h = self.input_mlp(h)

        # Create batched edge index for PyG
        batch_offset = torch.arange(batch_size, device=device).repeat_interleave(edge_index.size(1)) * num_nodes
        batched_edge_index = edge_index.repeat(1, batch_size) + batch_offset

        # Process through GATConv layers
        for conv, ln, proj in zip(self.conv_layers, self.layer_norms, self.projection_layers):
            h_prev = h
            h = conv(h, batched_edge_index, edge_features.reshape(-1, edge_features.size(-1)))
            h = proj(h)  # Project concatenated heads to match input dimension
            h = ln(h)
            h = F.relu(h)
            h = h + h_prev  # Residual connection

        # Reshape back to (batch_size, n_nodes, hid_dim)
        h = h.view(batch_size, num_nodes, -1)

        # Output layer
        output = self.output_layer(h)

        # Apply dropout if training
        if self.dropout is not None and self.training:
            output = self.dropout(output)

        return output

# ======= GATv2-based Recurrent Model =======
class GATv2RecurrentModel(nn.Module):
    def __init__(self, n_nodes, hid_dim, orbits, adj, num_layers=4, p_dropout=None, sparse=False):
        super(GATv2RecurrentModel, self).__init__()
        self.gat = GATv2Model(n_nodes, hid_dim, orbits, adj, num_layers=num_layers, p_dropout=p_dropout, sparse=False)

        # GRU-style hidden state update
        self.update_gate = nn.Linear(2 + hid_dim, hid_dim)
        self.reset_gate = nn.Linear(2 + hid_dim, hid_dim)
        self.candidate_state = nn.Linear(2 + hid_dim, hid_dim)

    def forward(self, x, hx):
        x = x.to(device)  
        hx = hx.to(device)  
        h = self.gat(x)  
        x_h_concat = torch.cat([h, hx], dim=-1)
        r = torch.sigmoid(self.reset_gate(x_h_concat))
        u = torch.sigmoid(self.update_gate(x_h_concat))
        c = torch.tanh(self.candidate_state(torch.cat([h, r * hx], dim=-1)))
        h_new = u * hx + (1 - u) * c
        return h, h_new


# ======= Encoder & Decoder =======
class GATv2Encoder(nn.Module):
    def __init__(self, n_nodes, hid_dim, orbits, adj, num_layers=1, sparse=False):
        super(GATv2Encoder, self).__init__()
        self.gat_layers = nn.ModuleList([
            GATv2RecurrentModel(n_nodes, hid_dim, orbits, adj, sparse=False)
        ])

    def forward(self, inputs, hidden_state):
        hidden_states = []
        output = inputs
        for layer_num, gat_layer in enumerate(self.gat_layers):
            output, next_hidden_state = gat_layer(output, hidden_state[layer_num])  
            hidden_states.append(next_hidden_state)

        return output, hidden_states[0]

class GATv2Decoder(nn.Module):
    def __init__(self, n_nodes, hid_dim, orbits, adj, num_layers=1, sparse=False):
        super(GATv2Decoder, self).__init__()
        self.gat_layers = nn.ModuleList([
            GATv2RecurrentModel(n_nodes, hid_dim, orbits, adj)
        ])

    def forward(self, inputs, hidden_state):
        hidden_states = []
        output = inputs
        for layer_num, gat_layer in enumerate(self.gat_layers):
            output, next_hidden_state = gat_layer(output, hidden_state[layer_num])  
            hidden_states.append(next_hidden_state)

        return output, hidden_states[0]

# ======= Seq2Seq Model =======
class GATv2Seq2SeqModel(nn.Module):
    def __init__(self, n_nodes, hid_dim, orbits, adj, num_layers=1, sparse=False):
        super(GATv2Seq2SeqModel, self).__init__()
        self.encoder = GATv2Encoder(n_nodes, hid_dim, orbits, adj, num_layers, sparse=False)
        self.decoder = GATv2Decoder(n_nodes, hid_dim, orbits, adj, num_layers, sparse=False)

    def forward(self, inputs, labels):
        """
        Args:
            inputs: (seq_len, batch_size, num_nodes, input_dim) - 3 timesteps of inputs
            labels: (horizon, batch_size, num_nodes, output_dim) - Expected outputs (1 timestep ahead)

        Returns:
            predictions: (horizon, batch_size, num_nodes, output_dim)
        """
        batch_size = inputs.shape[1]

        # Initialize hidden state (zero initially, will be updated across time)
        hidden_state = torch.zeros(
            1, batch_size, inputs.shape[2], self.encoder.gat_layers[0].gat.hid_dim, device=device
        )

        for t in range(inputs.shape[0]):  # Loop through time steps (e.g., 3 steps)
            _, hidden_state = self.encoder(inputs[t], hidden_state)

        decoder_input = torch.zeros_like(labels[0], device=device)  # Initial input to decoder (zeros)
        decoder_hidden_state = hidden_state
        outputs = []

        for t in range(labels.shape[0]):  
            output, decoder_hidden_state = self.decoder(decoder_input, decoder_hidden_state)
            outputs.append(output)
            decoder_input = output  

        return torch.stack(outputs)


