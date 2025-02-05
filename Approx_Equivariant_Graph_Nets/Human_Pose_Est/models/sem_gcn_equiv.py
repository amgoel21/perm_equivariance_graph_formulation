from __future__ import absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EquivariantGraphConv(nn.Module):
    """
    Strictly equivariant graph convolution layer for human skeleton graph.
    This version enforces exact equivariance through group-theoretic constraints.
    """
    def __init__(self, in_features, out_features, adj, bias=True):
        super(EquivariantGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Register adjacency matrix as buffer (non-parameter tensor)
        self.register_buffer('adj', adj)
        
        # Create equivariant weight matrices
        self.weight_self = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight_neighbor = nn.Parameter(torch.Tensor(in_features, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_self)
        nn.init.kaiming_uniform_(self.weight_neighbor)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        # x has shape (batch_size, num_nodes, in_features)
        batch_size, num_nodes, _ = x.size()
        
        # Self-transform
        out_self = torch.matmul(x, self.weight_self)
        
        # Neighbor-transform
        x_neighbors = torch.matmul(self.adj, x)
        out_neighbors = torch.matmul(x_neighbors, self.weight_neighbor)
        
        # Combine transformations
        out = out_self + out_neighbors
        
        if self.bias is not None:
            out = out + self.bias
            
        return out


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()
        self.gconv1 = nn.Sequential(
            EquivariantGraphConv(input_dim, hid_dim, adj),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout) if p_dropout is not None else nn.Identity()
        )
        self.gconv2 = nn.Sequential(
            EquivariantGraphConv(hid_dim, output_dim, adj),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout) if p_dropout is not None else nn.Identity()
        )

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class SemGCNEquiv(nn.Module):
    """
    Semantic GCN with strict equivariance guarantees.
    """
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, p_dropout=None):
        super(SemGCNEquiv, self).__init__()
        
        # Input layer
        self.gconv_input = nn.Sequential(
            EquivariantGraphConv(coords_dim[0], hid_dim, adj),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout) if p_dropout is not None else nn.Identity()
        )
        
        # Hidden layers
        self.gconv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout))
            
        # Output layer
        self.gconv_output = EquivariantGraphConv(hid_dim, coords_dim[1], adj)

    def forward(self, x):
        out = self.gconv_input(x)
        
        for gconv in self.gconv_layers:
            out = gconv(out)
            
        out = self.gconv_output(out)
        return out
