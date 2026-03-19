from data import set_seed, MonotoneConstRunDataset, SymDiffDataset, TargetSumDataset, MaxCyclicProductDataset, MaxCyclicAlterSumDataset
import torch

def _digits_from_tensor(x):
    return [str(int(v.item())) for v in x]

def _reverse_1d(t):
    return torch.flip(t, dims=[0])

def check_equivariant_unique_longest(dataset):
    """
    For equivariant datasets:
      - label == unique-longest mask if its length >= run_length, else all-zeros
      - reversal equivariance holds for ground-truth masks
      - no ties slip through
    """
    n = min(300, len(dataset))
    for i in range(n):
        x, y = dataset[i]
        if i % 5 == 0:
            print(x, y)
        seq = _digits_from_tensor(x)
        mask, ok = dataset._gt_mask_ok(seq)
        if not ok:
            return False, f"tie detected at idx {i} (should be regenerated)"
        L = sum(mask)
        expected = mask if L >= dataset.run_length else [0]*len(mask)
        if y.dtype != torch.long:
            return False, f"label dtype not long at idx {i}"
        if y.tolist() != expected:
            return False, f"mask mismatch at idx {i}: expected={expected}, got={y.tolist()}"

        # Check reversal on ground-truth
        xr = _reverse_1d(x)
        seq_r = _digits_from_tensor(xr)
        mask_r, ok_r = dataset._gt_mask_ok(seq_r)
        if not ok_r:
            return False, f"tie after reversal at idx {i}"
        if mask_r != list(reversed(mask)):
            return False, f"reversal mismatch at idx {i}: gt_rev={mask_r}, should={list(reversed(mask))}"
    return True, "equivariant checks passed"

def check_invariant_unique_longest(dataset):
    """
    For invariant datasets: scalar y == 1 iff unique-longest length >= run_length, else 0.
    """
    n = min(300, len(dataset))
    for i in range(n):
        x, y = dataset[i]
        if i == 0:
            print(x, y)
        seq = _digits_from_tensor(x)
        mask, ok = dataset._gt_mask_ok(seq)
        if not ok:
            return False, f"tie detected at idx {i} (should be regenerated)"
        L = sum(mask)
        expected = 1 if L >= dataset.run_length else 0
        if int(y.item()) != expected:
            return False, f"scalar label mismatch at idx {i}: expected={expected}, got={int(y.item())}"
    return True, "invariant checks passed"

# ---- quick demo ----
num_samples = 5
max_prod = MaxCyclicProductDataset(num_samples=num_samples, seq_length=6, cyc_length=3, vocab_size=10, seed=42)
max_alt_sum = MaxCyclicAlterSumDataset(num_samples=num_samples, seq_length=6, cyc_length=3, vocab_size=10, seed=42)
for i in range(num_samples):
    print(max_prod[i])
for i in range(num_samples):
    print(max_alt_sum[i])

sym_diff = SymDiffDataset(num_samples=5, seq_length=8, seed=42, equivariant=True)
for i in range(num_samples):
    print(sym_diff[i])

target_sum = 3.0
ds_eq = TargetSumDataset(num_samples=5, seq_length=8, seed=123,
                             equivariant=True, vocab_size=4, target_sum=target_sum)
for i in range(num_samples):
    print(ds_eq[i])

# Monotone, non-strict
mono_eq = MonotoneConstRunDataset(num_samples=40, seq_length=8, run_length=4, seed=1,
                                  equivariant=True, strict=True, equal=False)
print(check_equivariant_unique_longest(mono_eq))
mono_inv = MonotoneConstRunDataset(num_samples=40, seq_length=8, run_length=4, seed=2,
                                  equivariant=False, strict=True, equal=False)
print(check_invariant_unique_longest(mono_inv))

# Constant
const_eq = MonotoneConstRunDataset(num_samples=40, seq_length=8, run_length=4, seed=3,
                                   equivariant=True, equal=True)
print(check_equivariant_unique_longest(const_eq))
const_inv = MonotoneConstRunDataset(num_samples=40, seq_length=8, run_length=4, seed=4,
                                   equivariant=False, equal=True)
print(check_invariant_unique_longest(const_inv))