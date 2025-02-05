
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import numpy as np
from common.utils import orbit_2d, sparsify_corrected
import math

class EdgeEmbedder(nn.Module):
    def __init__(self, num_categories, embedding_dim, category_norms):
        """
        Initialize the edge embedder with category-specific max norms.
        Args:
            num_categories: Number of unique edge categories.
            embedding_dim: Dimension of the embedding vectors.
            category_norms: Dictionary mapping category indices to their max norms.
        """
        super(EdgeEmbedder, self).__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim, max_norm=None)
        self.category_norms = category_norms

    def forward(self, category_indices):
        """
        Retrieve embeddings with category-specific max norms.
        Args:
            category_indices: Tensor of category indices.
        Returns:
            Tensor of embeddings with adjusted norms.
        """
        embeddings = self.embedding(category_indices)
        # Apply max_norm for each category dynamically
        norms = torch.tensor(
            [self.category_norms[cat.item()] for cat in category_indices], dtype=torch.float
        ).to(embeddings.device).unsqueeze(1)
        # Compute the current norms of the embeddings
        current_norms = torch.norm(embeddings, dim=1, keepdim=True)
        # Avoid division by zero by replacing zeros with a small positive value
        current_norms = torch.where(current_norms == 0, torch.tensor(1e-8, device=embeddings.device), current_norms)
        # Scale only where the current norm exceeds the specified norm
        scale_factors = torch.min(torch.ones_like(current_norms), norms / current_norms)
        embeddings = embeddings * scale_factors
        return embeddings




class GATv2Model(nn.Module):
    """
    GAT-based model with permutation-aware edge features and concatenated attention heads.
    """
    def __init__(self, n_nodes, hid_dim, perms, coords_dim=(2, 3), num_layers=4, p_dropout=None, sparse=False, maxnorm=float('inf'), soft = False):
        super(GATv2Model, self).__init__()
        self.n = n_nodes  # number of nodes
        self.hid_dim = hid_dim

        if sparse:
            # Use sparsify_corrected to handle main and non-main edges
            self.num_categories, edge_index, edge_categories, category_norms = sparsify_corrected(perms, max_norm_main=maxnorm, soft = soft)
            self.edge_index = torch.tensor(edge_index).t().contiguous()
            self.edge_categories = torch.tensor(edge_categories)
            self.edge_embedder = EdgeEmbedder(self.num_categories, embedding_dim=hid_dim, category_norms=category_norms)
        else:
            # Fully connected graph
            edge_indices = [(i, j) for i in range(self.n) for j in range(self.n)]
            self.edge_orbits = orbit_2d(perms)
            self.num_categories = len(self.edge_orbits)

            # Create category mapping for each edge
            self.edge_to_category = {}
            for cat_idx, orbit in enumerate(self.edge_orbits):
                for edge in orbit:
                    self.edge_to_category[edge] = cat_idx

            # Edge embedder
            self.edge_embedder = EdgeEmbedder(self.num_categories, embedding_dim=hid_dim, category_norms={cat_idx: 1 for cat_idx in range(self.num_categories)})

            # Create edge index and category tensors for PyG (fully connected)
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




    


    