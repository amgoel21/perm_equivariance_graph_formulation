from Multi_GAT import index, find_categories, compute_closure
import math
import torch
import torch.nn as nn

# ---------- Baseline classes using the helpers ----------
def _build_pair_orbits_with_helpers(n, *, perms=None, orbits=None, adj=None, sparse=False):
    num_cats, edge_index, edge_categories = compute_closure(
        n, perms=perms, orbits=orbits, adj=adj, sparse=sparse
    )
    param_idx = torch.full((n, n), -1, dtype=torch.long)
    param_idx[edge_index[0], edge_index[1]] = edge_categories
    if (param_idx < 0).any():
        raise ValueError("Incomplete category assignment: some (i,j) missing.")
    return param_idx, num_cats

def _build_output_orbits_with_helpers(n, *, orbits=None):
    if orbits is None:
        return None, 0
    bias_idx = torch.tensor(index(n, orbits), dtype=torch.long)  # (n,)
    return bias_idx, len(orbits)

class PermEquivariantLinear(nn.Module):
    def __init__(self, n, d_in, d_out, *, perms=None, orbits=None, adj=None,
                 sparse=False, use_bias=True, separable=True):
        super().__init__()
        self.n, self.d_in, self.d_out = n, d_in, d_out
        self.separable = separable

        param_idx, n_params = _build_pair_orbits_with_helpers(
            n, perms=perms, orbits=orbits, adj=adj, sparse=sparse
        )
        self.register_buffer("param_idx", param_idx)  # (n,n)

        if separable:
            # K is scalar per 2-orbit; channels mixed by one global W_feat: R^{d_in} \to \R^{d_out}
            self.alpha = nn.Parameter(torch.randn(n_params) / math.sqrt(n))
            self.W_feat = nn.Parameter(torch.empty(d_in, d_out))
            nn.init.xavier_uniform_(self.W_feat)
        else:
            # Full tensor per 2-orbit: shape (#2-orbits, d_in, d_out) (i.e., d_in*d_out copies of weights)
            self.alpha_full = nn.Parameter(torch.empty(n_params, d_in, d_out))
            nn.init.xavier_uniform_(self.alpha_full)

        bias_idx, n_bias = _build_output_orbits_with_helpers(n, orbits=orbits) if use_bias else (None, 0)
        if bias_idx is not None:
            self.register_buffer("bias_idx", bias_idx)        # (n,)
            self.bias = nn.Parameter(torch.zeros(n_bias, d_out))
        else:
            self.bias_idx = None
            self.register_parameter("bias", None)

    def forward(self, x):  # x: (B, n, d_in)
        B, n, din = x.shape
        assert n == self.n and din == self.d_in

        if self.separable:
            # x_proj: (B, n, d_out)
            x_proj = x @ self.W_feat
            # K: (n,n)
            K = self.alpha[self.param_idx]
            # y: (B, n, d_out)
            y = torch.einsum("ij,bjd->bid", K, x_proj)
        else:
            # Look up a (d_in,d_out) matrix per (i,j) via 2-orbit id
            # M: (n,n,d_in,d_out)
            M = self.alpha_full[self.param_idx]  # broadcast gather
            # y_i = sum_j x_j @ M[i,j]
            # (B,n,d_in) x (n,n,d_in,d_out) -> (B,n,d_out)
            y = torch.einsum("bjd,ijdo->bio", x, M)

        if self.bias is not None:
            y = y + self.bias[self.bias_idx].unsqueeze(0)
        return y


class PermEquivariantMLP(nn.Module):
    def __init__(self, n, depth, d_in, d_out, separable=True,
                 nonlinearity=None, layer_specs=None, use_bias=True):
        super().__init__()
        L = depth
        if layer_specs is None:
            layer_specs = [{} for _ in range(L)]
        layers = []
        act = nonlinearity if nonlinearity is not None else nn.ReLU()
        for ell in range(L):
            specs = layer_specs[ell]
            layers.append(
                PermEquivariantLinear(
                    n, d_in, d_out,
                    perms=specs.get("perms"),
                    orbits=specs.get("orbits"),
                    adj=specs.get("adj"),
                    sparse=specs.get("sparse", False),
                    use_bias=use_bias,
                )
            )
            if ell < L - 1:
                layers.append(act)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------- Palindrome subgroup on n=5 ----------
n = 5
d_in, d_out = 1, 1
separable = True

# Node orbits under reflection i -> 4 - i: {0,4}, {1,3}, {2}
node_orbits = [[0,4], [1,3], [2]]

# Build a 2-layer equivariant MLP as baseline
model = PermEquivariantMLP(
    n=n,
    depth=2,
    d_in=d_in,
    d_out=d_out,
    separable=separable,
    nonlinearity=nn.ReLU(),  # any pointwise nonlinearity preserves equivariance
    layer_specs=[{"orbits": node_orbits}, {"orbits": node_orbits}],
    use_bias=True
)

# Create the permutation matrix for reflection
perm = torch.tensor([4, 3, 2, 1, 0], dtype=torch.long)  # i -> 4 - i
P = torch.zeros(n, n)
P[torch.arange(n), perm] = 1.0

# Synthetic input batch
B = 7
x = torch.randn(B, n, d_in)

# Check equivariance: f(Px) == P f(x)
with torch.no_grad():
    y1 = model(x)
    Px = x[:, perm, :]
    y2 = model(Px)           # f(Px)
    y1_perm = y1[:, perm, :]           # P f(x)

err = (y2 - y1_perm).abs().max().item()

# first layer weight sharing error
first = model.net[0]
param_idx = first.param_idx            # (n, n)
perm_idx  = perm                       # LongTensor of shape (n,)

if hasattr(first, "alpha"):  # separable case: K = alpha[param_idx], features via W_feat
    # K[i,j] = alpha[orbit(i,j)]
    K = first.alpha[param_idx].detach()                     # (n, n)
    K_perm = K[perm_idx][:, perm_idx]                      # K after node permutation
    max_tie_err = (K - K_perm).abs().max().item()

elif hasattr(first, "alpha_full"):  # non-separable: per-orbit (d_in x d_out) matrix
    # M[i,j,:,:] = alpha_full[orbit(i,j), :, :]
    M = first.alpha_full[param_idx].detach()               # (n, n, d_in, d_out)
    M_perm = M[perm_idx][:, perm_idx]                      # permute both node axes
    max_tie_err = (M - M_perm).abs().max().item()

else:
    raise RuntimeError("Unknown layer parameterization.")

# Report
print("Max equivariance error ||f(Px) - P f(x)||_∞:", err)
print("Max parameter-tying error on W under reflection:", max_tie_err)

# Show category matrix (2-orbit ids) for transparency
cat_idx = first.param_idx
print("\nEdge category matrix (2-orbit ids):\n", cat_idx)