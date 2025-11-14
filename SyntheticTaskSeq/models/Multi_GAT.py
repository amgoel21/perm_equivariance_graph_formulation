from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import numpy as np
from sympy import *
from sympy.combinatorics import Permutation, PermutationGroup
from random import randrange
import math


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
        # nn.init.kaiming_uniform_(self.embedding.weight, a=0.3)

    def forward(self, category_indices):
        """
        Retrieve embeddings with category-specific max norms.
        Args:
            category_indices: Tensor of category indices.
        Returns:
            Tensor of embeddings with adjusted norms.
        """
        embeddings = self.embedding(category_indices)
        return embeddings


def index(n, orbits):
    """
    Group nodes by orbit
    """
    pos = [0]*n
    for a in range(len(orbits)):
        for b in orbits[a]:
            pos[b]=a
    return pos

def sparsify(n, pos,adj,orbits):
    """
    Create 2-closure for sparse graph
    Given disjoint node orbits and adjacency
    Returns number of edge categories, edge index, and edge categories
    """
    edge_index=[]
    edge_categories=[[] for i in range(len(orbits)+len(orbits)**2)]
    for i in range(n):
        for j in range(n):
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

def find_categories(n_nodes, pos,orbits):
    """
    For fully connected graph, returns edge categories
    """
    num_orbits = len(orbits)
    edge_categories=[[] for i in range(num_orbits+num_orbits**2)]
    for i in range(n_nodes):
        for j in range(n_nodes):
            if(i==j):
                i_orbit = pos[i]
                edge_categories[num_orbits**2+i_orbit].append((i,j))
            else:
                i_orbit = pos[i]
                j_orbit = pos[j]
                edge_categories[i_orbit*num_orbits+j_orbit].append((i,j))
    return edge_categories

def compute_closure_orbits(n, orbits, adj = None, sparse = False):
    """
    Returns 2-closure information of graph structure
    """
    pos = index(n, orbits)
    if sparse:
        num_categories, edge_index, edge_orbits, = sparsify(n, pos,adj,orbits)
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
        edge_orbits = find_categories(n, pos,orbits)
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

def compute_closure_perms(n, perms):
    """
    Returns 2-closure information from permutation generators of equivariance group
    """
    edge_indices = [(i, j) for i in range(n) for j in range(n)]
    edge_orbits = orbit_2d(perms)
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

def compute_closure(n, perms = None, orbits = None, adj = None, sparse = False):
    if perms is not None:
        return compute_closure_perms(n, perms)
    elif orbits is not None:
        return compute_closure_orbits(n, orbits, adj, sparse)
    return False

def perm_2d(perm):
    # Turns a permutation into permutation of 2d orbits
    perm_2d=[0]*(len(perm))**2
    for i in range(len(perm)):
        for j in range(len(perm)):
            perm_2d[i*len(perm)+j]=len(perm)*perm[i]+perm[j]
    return perm_2d

def orbit_2d(constructors):
    # Returns 2d orbit of group from list of constructors
    n = constructors[0].size
    perms_2d=[]
    for c in constructors:
        perms_2d.append(Permutation(perm_2d(list(c))))
    new_group=PermutationGroup(perms_2d)
    orbits = new_group.orbits()
    altered_orbits = []
    for orbit in orbits:
        new_orbit = []
        for i in orbit:
            new_orbit.append((i//n,i%n))
        altered_orbits.append(new_orbit)
    return altered_orbits



    
    

class GATv2Model(nn.Module):
    """
    GAT-based model with permutation-aware edge features and concatenated attention heads.
    """
    def __init__(self, n_nodes, hid_dim, orbits, adj, coords_dim=(2, 3), num_layers=4, p_dropout=None, sparse=False):
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
        self.output_layer = nn.Linear(hid_dim, 1)

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
        h_agg = torch.sum(h, dim=1)

        # Output layer
        output = self.output_layer(h_agg)

        # Apply dropout if training
        if self.dropout is not None and self.training:
            output = self.dropout(output)

        return output



class TokenEmbedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TokenEmbedder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


class MultiGraphGATv2Model_inv(nn.Module):
    def __init__(self, graph_configs, hid_dim, num_layers=4, p_dropout=0.05, vocab_size=20):
        super(MultiGraphGATv2Model_inv, self).__init__()

        self.graph_configs = graph_configs
        self.hid_dim = hid_dim
        self.shared_num_heads = 8
        self.shared_head_dim = hid_dim // self.shared_num_heads


        self.token_embedders = nn.ModuleDict()

        self.edge_indices = {}
        self.num_categories= {}
        self.edge_categories = {}
        # self.edge_embedders = {}
        self.edge_embedders = nn.ModuleDict()
        self.output_dims = {}

        for struct_id, config in graph_configs.items():
            n_nodes, perms, adj, coords_dim, orbits, sparse, out_dim = config["n_nodes"], config['perms'], config["adj"], config["coords_dim"], config["orbits"], config["sparse"], config["out_dim"]

            num_categories, edge_index, edge_categories = compute_closure(n_nodes, perms, orbits, adj, sparse)

            self.edge_indices[struct_id] = edge_index
            self.edge_categories[struct_id] = edge_categories
            self.num_categories[struct_id] = num_categories
            self.edge_embedders[struct_id] = EdgeEmbedder(num_categories, embedding_dim=hid_dim)

            
            self.token_embedders[struct_id] = TokenEmbedder(vocab_size, hid_dim)

            self.output_dims[struct_id] = out_dim

        self.max_output_dim = max(self.output_dims.values())

        # self.input_mlp = nn.Sequential(
        #     nn.Linear(1, hid_dim),
        #     nn.LayerNorm(hid_dim),
        #     nn.ReLU(),
        #     nn.Linear(hid_dim, hid_dim),
        #     nn.LayerNorm(hid_dim)
        # )
        
        self.input_mlp = nn.Sequential(
            nn.Linear(1, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU()
        )

        # self.output_layer = nn.Linear(hid_dim, self.max_output_dim)
        self.output_layer = nn.Linear(hid_dim, 1)

        self.conv_layers = nn.ModuleList([
            GATv2Conv(hid_dim, self.shared_head_dim, edge_dim=hid_dim, heads=self.shared_num_heads, concat=True)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hid_dim) for _ in range(num_layers)])
        self.projection_layers = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for _ in range(num_layers)])

        self.dropout = nn.Dropout(p_dropout) if p_dropout is not None else None

    def forward(self, x, structure_ids):
        """
        x: (batch_size, num_nodes, 1)
        structure_ids: list of structure_id strings (len = batch_size)
        """
        device = x.device
        batch_size, num_nodes, _ = x.size()
        outputs = []
    
        for i in range(batch_size):
            struct_id = structure_ids[i]
            edge_index = self.edge_indices[struct_id].to(device)
            edge_embedder = self.edge_embedders[struct_id].to(device)
            token_embedder = self.token_embedders[struct_id].to(device)
    
            edge_feat = edge_embedder(self.edge_categories[struct_id].to(device)).unsqueeze(0)
            node_indices = x[i].squeeze(-1).long()  # (num_nodes,)
            h = self.input_mlp(node_indices.unsqueeze(-1).float())
            # h = token_embedder(node_indices.unsqueeze(0))
            # h = self.input_mlp(h)
            h = h.squeeze(0)
            # h = token_embedder(node_indices.unsqueeze(0))  # (1, num_nodes, hid_dim)
            # h = self.input_mlp(h)  # (1, num_nodes, hid_dim)
            # h = h.squeeze(0)
            edge_feat = edge_feat.squeeze(0)
    
            for conv, ln, proj in zip(self.conv_layers, self.layer_norms, self.projection_layers):
                h_prev = h
                h = conv(h, edge_index, edge_feat)
                h = proj(h)
                h = ln(h)
                h = F.relu(h)
                h = h + h_prev
    
            h = h.sum(dim=0, keepdim=True)  # (1, hid_dim)
            out = self.output_layer(h)  # (1, max_output_dim)
    
            if self.dropout is not None and self.training:
                out = self.dropout(out)
    
            outputs.append(out)
    
        return torch.cat(outputs, dim=0)  # (batch_size, max_output_dim)
    @torch.no_grad()
    def edge_features_for_pairs(self, structure_id: str, pairs: torch.LongTensor):
        """
        Return learned edge embeddings for each (i,j) in `pairs` for the given structure.
    
        Args:
            structure_id: key in self.graph_configs / self.edge_* dicts
            pairs: LongTensor [N,2] of node index pairs (ordered; may include self-pairs)
    
        Returns:
            Dict[str, Tensor]: {"edge": Tensor [N, hid_dim]}
        """
        device = next(self.parameters()).device
        edge_index = self.edge_indices[structure_id].to(device)        # [2, E]
        edge_cats  = self.edge_categories[structure_id].to(device)     # [E]
        embedder   = self.edge_embedders[structure_id].to(device)      # nn.Embedding or wrapper
    
        # Build (i,j) -> category id map
        edge_map = { (int(edge_index[0, e]), int(edge_index[1, e])): int(edge_cats[e])
                     for e in range(edge_index.size(1)) }
    
        rows = []
        for i, j in pairs.tolist():
            if (i, j) in edge_map:
                cat = edge_map[(i, j)]
                rows.append(embedder(torch.tensor([cat], device=device)).squeeze(0))
            else:
                # If the directed edge (i,j) is not present, emit a zero vector
                rows.append(torch.zeros(self.hid_dim, device=device))
    
        feats = torch.stack(rows, dim=0)  # [N, hid_dim]
        return {"edge": feats}
    @torch.no_grad()
    def dump_edge_features(self, out_dir: str, tag: str = "", structures: list = None, include_all_pairs: bool = True):
        # Local imports so this works even if the module forgot them
        import os, json
        import numpy as np
        import torch
    
        os.makedirs(out_dir, exist_ok=True)
        device = next(self.parameters()).device
        structures = list(self.graph_configs.keys()) if structures is None else structures
    
        for struct_id in structures:
            cfg = self.graph_configs[struct_id]
            n_nodes = int(cfg["n_nodes"] if isinstance(cfg, dict) else cfg.n_nodes)
    
            if include_all_pairs:
                I, J = torch.meshgrid(
                    torch.arange(n_nodes, device=device),
                    torch.arange(n_nodes, device=device),
                    indexing="ij",
                )
                pairs = torch.stack([I.reshape(-1), J.reshape(-1)], dim=-1)  # [N,2]
            else:
                pairs = self.edge_indices[struct_id].to(device).t()  # [E,2]
    
            feat_dict = self.edge_features_for_pairs(struct_id, pairs)  # {"edge": [N, hid_dim]}
            if isinstance(feat_dict, torch.Tensor):
                feat_dict = {"edge": feat_dict}
    
            np_pairs = pairs.detach().cpu().numpy()
            save_dict = {"pairs": np_pairs}
            layer_names = []
            for k, v in feat_dict.items():
                layer_names.append(k)
                save_dict[k] = v.detach().cpu().numpy()
    
            tag_sfx = f"__{tag}" if tag else ""
            base = os.path.join(out_dir, f"edge_feats__{struct_id}{tag_sfx}")
    
            np.savez_compressed(base + ".npz", **save_dict)
            with open(base + ".meta.json", "w") as f:
                json.dump(
                    {
                        "structure": struct_id,
                        "n_nodes": n_nodes,
                        "num_pairs": int(np_pairs.shape[0]),
                        "layers": layer_names,
                        "arrays": list(save_dict.keys()),
                        "all_pairs": bool(include_all_pairs),
                    },
                    f,
                    indent=2,
                )
    
            print(f"[dump_edge_features(inv)] wrote {base}.npz (pairs={np_pairs.shape[0]}, layers={layer_names})")

    






class MultiGraphGATv2Model_equiv(nn.Module):
    """
    GAT-based model that supports multiple equivariant input structures.
    Each structure has its own token embedding and per-node output head.
    """
    def __init__(self, graph_configs, hid_dim, num_layers=4, p_dropout=0.05, vocab_size=20):
        super(MultiGraphGATv2Model_equiv, self).__init__()

        self.graph_configs = graph_configs
        self.hid_dim = hid_dim
        self.shared_num_heads = 8
        self.shared_head_dim = hid_dim // self.shared_num_heads

        self.token_embedders = nn.ModuleDict()
        self.edge_indices = {}
        self.num_categories= {}
        self.edge_categories = {}
        # self.edge_embedders = {}
        self.edge_embedders = nn.ModuleDict()
        self.output_dims = {}

        for struct_id, config in graph_configs.items():
            n_nodes, perms, adj, coords_dim, orbits, sparse, out_dim = config["n_nodes"], config['perms'], config["adj"], config["coords_dim"], config["orbits"], config["sparse"], config["out_dim"]

            num_categories, edge_index, edge_categories = compute_closure(n_nodes, perms, orbits, adj, sparse)

            self.edge_indices[struct_id] = edge_index
            self.edge_categories[struct_id] = edge_categories
            self.num_categories[struct_id] = num_categories
            self.edge_embedders[struct_id] = EdgeEmbedder(num_categories, embedding_dim=hid_dim)
            self.token_embedders[struct_id] = TokenEmbedder(vocab_size, hid_dim)
            self.output_dims[struct_id] = out_dim

        self.input_mlp = nn.Sequential(
            nn.LayerNorm(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim)
        )

        self.conv_layers = nn.ModuleList([
            GATv2Conv(hid_dim, self.shared_head_dim, edge_dim=hid_dim, heads=self.shared_num_heads, concat=True)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hid_dim) for _ in range(num_layers)])
        self.projection_layers = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for _ in range(num_layers)])

        self.output_layer = nn.Linear(hid_dim, 2)

        self.dropout = nn.Dropout(p_dropout) if p_dropout is not None else None

    def forward(self, x, structure_ids):
        """
        x: (batch_size, num_nodes, 1)
        structure_ids: list of strings of length batch_size
        """
        device = x.device
        batch_size, num_nodes, _ = x.size()
        outputs = []
    
        for i in range(batch_size):
            struct_id = structure_ids[i]
            edge_index = self.edge_indices[struct_id].to(device)
            edge_embedder = self.edge_embedders[struct_id].to(device)
            token_embedder = self.token_embedders[struct_id].to(device)
    
            edge_feat = edge_embedder(self.edge_categories[struct_id].to(device)).unsqueeze(0)  # (1, num_edges, hid_dim)
            node_indices = x[i].squeeze(-1).long()  # (num_nodes,)
            h = token_embedder(node_indices.unsqueeze(0))  # (1, num_nodes, hid_dim)
            h = self.input_mlp(h)  # (1, num_nodes, hid_dim)
            # h = self.input_mlp(node_indices.unsqueeze(-1).float())
    
            h = h.squeeze(0)  # (num_nodes, hid_dim)
            edge_feat = edge_feat.squeeze(0)  # (num_edges, hid_dim)
    
            for conv, ln, proj in zip(self.conv_layers, self.layer_norms, self.projection_layers):
                h_prev = h
                h = conv(h, edge_index, edge_feat)
                h = proj(h)
                h = ln(h)
                h = F.relu(h)
                h = h + h_prev  # residual
    
            out = self.output_layer(h)  # (num_nodes, out_dim)
            if self.dropout is not None and self.training:
                out = self.dropout(out)
    
            outputs.append(out.unsqueeze(0))  # (1, num_nodes, out_dim)
    
        return torch.cat(outputs, dim=0)  # (batch_size, num_nodes, out_dim)
    
    @torch.no_grad()
    def edge_features_for_pairs(self, structure_id: str, pairs: torch.LongTensor):
        """
        Return learned edge embeddings for each (i,j) in `pairs` for the given structure.
    
        Args:
            structure_id: key in self.graph_configs / self.edge_* dicts
            pairs: LongTensor [N,2] of node index pairs (ordered; includes self-pairs if you pass them)
    
        Returns:
            Dict[str, Tensor]: {"edge": Tensor [N, hid_dim]}
        """
        import torch
        device = next(self.parameters()).device if any(True for _ in self.parameters()) else torch.device("cpu")
    
        edge_index = self.edge_indices[structure_id].to(device)        # [2, E]
        edge_cats  = self.edge_categories[structure_id].to(device)     # [E]
        embedder   = self.edge_embedders[structure_id].to(device)      # nn.Embedding or wrapper
    
        # Build (i,j) -> category id lookup
        edge_map = { (int(edge_index[0, e]), int(edge_index[1, e])): int(edge_cats[e])
                     for e in range(edge_index.size(1)) }
    
        # For each requested (i,j), embed if present; else zero vector
        feat_rows = []
        for i, j in pairs.tolist():
            if (i, j) in edge_map:
                cat = edge_map[(i, j)]
                feat_rows.append(embedder(torch.tensor([cat], device=device)).squeeze(0))
            else:
                feat_rows.append(torch.zeros(self.hid_dim, device=device))
    
        feats = torch.stack(feat_rows, dim=0)  # [N, hid_dim]
        return {"edge": feats}
    @torch.no_grad()
    def dump_edge_features(self, out_dir: str, tag: str = "", structures: list = None, include_all_pairs: bool = True):
        """
        Dump edge features to disk for each dataset/structure in this model.
    
        Writes, per structure:
          out_dir/
            edge_feats__{structure}__{tag}.npz        (arrays: pairs [N,2], edge [N,hid_dim])
            edge_feats__{structure}__{tag}.meta.json  (small metadata JSON)
    
        Args:
            out_dir: directory to write files into
            tag: optional suffix to distinguish runs (e.g., 'multitask__mix=...' or 'trial=1')
            structures: optional subset list of structure_ids to dump; default = all
            include_all_pairs: if True, dumps ALL ordered (i,j) pairs (including self-edges).
                               If False, dumps only the edges present in self.edge_indices[struct].
        """
        import os, json
        import numpy as np
        import torch
    
        os.makedirs(out_dir, exist_ok=True)
        device = next(self.parameters()).device if any(True for _ in self.parameters()) else torch.device("cpu")
        structures = list(self.graph_configs.keys()) if structures is None else structures
    
        for struct_id in structures:
            cfg = self.graph_configs[struct_id]
            # support dict configs
            n_nodes = int(cfg["n_nodes"] if isinstance(cfg, dict) else cfg.n_nodes)
    
            if include_all_pairs:
                I, J = torch.meshgrid(
                    torch.arange(n_nodes, device=device),
                    torch.arange(n_nodes, device=device),
                    indexing="ij",
                )
                pairs = torch.stack([I.reshape(-1), J.reshape(-1)], dim=-1)  # [N,2] for all (i,j)
            else:
                # only the model's defined directed edges
                ei = self.edge_indices[struct_id].to(device)  # [2, E]
                pairs = ei.t()  # [E, 2]
    
            feat_dict = self.edge_features_for_pairs(struct_id, pairs)  # {"edge": [N, hid_dim]} or similar
    
            # normalize to dict[str->tensor]
            if isinstance(feat_dict, torch.Tensor):
                feat_dict = {"edge": feat_dict}
    
            np_pairs = pairs.detach().cpu().numpy()
            save_dict = {"pairs": np_pairs}
            layer_names = []
            for k, v in feat_dict.items():
                layer_names.append(k)
                save_dict[k] = v.detach().cpu().numpy()
    
            tag_sfx = f"__{tag}" if tag else ""
            base = os.path.join(out_dir, f"edge_feats__{struct_id}{tag_sfx}")
    
            # save arrays
            np.savez_compressed(base + ".npz", **save_dict)
            # save metadata
            with open(base + ".meta.json", "w") as f:
                json.dump(
                    {
                        "structure": struct_id,
                        "n_nodes": n_nodes,
                        "num_pairs": int(np_pairs.shape[0]),
                        "layers": layer_names,
                        "arrays": list(save_dict.keys()),
                        "all_pairs": bool(include_all_pairs),
                    },
                    f,
                    indent=2,
                )
    
            print(f"[dump_edge_features] wrote {base}.npz  (pairs={np_pairs.shape[0]}, layers={layer_names})")




        
        



class PermutationMLP(nn.Module):
    """
    Pipeline:
      x: (B, L, 1) →
        per-node Linear(1→H), ReLU → (B, L, H)
        token-mixing Linear over positions L→L (channel-wise) → (B, L, H)
        per-node Linear(H→2) → (B, L, 2)
        sum over positions → (B, 2)
    """
    def __init__(self, seq_length: int, hidden_dim: int = 128):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

        # Per-node feature lift: 1 → H
        self.in_proj = nn.Linear(1, hidden_dim)

        # Token-mixing across positions: L → L (applied channel-wise)
        # We apply this to the "positions" dimension; no bias needed.
        self.token_mixer = nn.Linear(seq_length, seq_length, bias=False)

        # Per-node head: H → 2
        self.out_proj = nn.Linear(hidden_dim, 2)

        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, L) or (B, L, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, L, 1)
        assert x.dim() == 3 and x.size(-1) == 1, f"Expected (B, L, 1), got {x.shape}"
        assert x.size(1) == self.seq_length, f"Expected seq_length={self.seq_length}, got {x.size(1)}"

        # Per-node lift
        z = self.act(self.in_proj(x))          # (B, L, H)

        # Token mixing across positions: apply Linear(L→L) per feature channel
        # Move L to last dim, mix, then move back
        z = z.transpose(1, 2)                  # (B, H, L)
        z = self.token_mixer(z) 
        z = self.act(z)
        z = z.transpose(1, 2)                  # (B, L, H)

        # Per-node logits and sum over positions
        per_pos_logits = self.out_proj(z)      # (B, L, 2)
        logits = per_pos_logits.sum(dim=1)     # (B, 2)
        return logits





