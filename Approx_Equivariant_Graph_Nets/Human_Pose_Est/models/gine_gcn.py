import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from common.utils import orbit_2d, sparsify_corrected


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


class GINEGCN(nn.Module):
    """
    GINEConv-based model with permutation-aware edge features.
    """
    def __init__(self, n_nodes, hid_dim, perms, coords_dim=(2, 3), num_layers=4, p_dropout=None, sparse=False,maxnorm=20):
        super(GINEGCN, self).__init__()
        self.n = n_nodes  # Number of nodes
        self.hid_dim = hid_dim

        if sparse:
            # Use sparsify_corrected to handle main and non-main edges
            self.num_categories, edge_index, edge_categories, category_norms = sparsify_corrected(perms,max_norm_main=maxnorm)
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

        # GINEConv layers
        self.input_mlp = nn.Sequential(
            nn.Linear(coords_dim[0], hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim)
        )

        conv_layers = []
        layer_norms = []
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                nn.LayerNorm(hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, hid_dim),
                nn.LayerNorm(hid_dim)
            )
            conv = GINEConv(mlp, edge_dim=hid_dim, train_eps=True)
            conv_layers.append(conv)
            layer_norms.append(nn.LayerNorm(hid_dim))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.layer_norms = nn.ModuleList(layer_norms)

        # Output layer
        self.output_layer = nn.Linear(hid_dim, coords_dim[1])

        # Dropout
        self.dropout = nn.Dropout(p_dropout) if p_dropout is not None else None

    def forward(self, x):
        device = x.device

        def single_item_forward(item):
            # Create edge features using embeddings with max norms
            edge_features = self.edge_embedder(self.edge_categories.to(device))

            # Initial feature transformation
            h = self.input_mlp(item)

            # Apply GINEConv layers with residual connections and layer norm
            for conv, ln in zip(self.conv_layers, self.layer_norms):
                identity = h
                h = conv(h, self.edge_index.to(device), edge_features)
                h = ln(h)
                h = F.relu(h)
                h = h + identity  # Residual connection

            # Output transformation
            return self.output_layer(h)

        # Process the batch in parallel using vmap
        output = torch.vmap(single_item_forward)(x)

        # Apply dropout after vmap if needed
        if self.dropout is not None and self.training:
            output = self.dropout(output)

        return output

    def reset_parameters(self):
        for conv in self.conv_layers:
            conv.reset_parameters()
        self.edge_embedder.embedding.reset_parameters()
        self.input_mlp[0].reset_parameters()
        self.input_mlp[3].reset_parameters()
        self.output_layer.reset_parameters()
        for ln in self.layer_norms:
            ln.reset_parameters()
