import torch
from torch.utils.data import Dataset, DataLoader
import random
import string
import argparse


def set_seed(seed=99):
    random.seed(seed)
    torch.manual_seed(seed)

class IsBalancedParenthesisDataset(Dataset):
    '''
    Given a string containing characters in '()[]{}', determine if it has balanced parentheses
    Example: input = '[()]' (mapped to integer index for torch dataset), label = 1
    '''
    def __init__(self, num_samples=10000, seq_length=10, seed=42):
        set_seed(seed)
        assert seq_length % 2 == 0, 'must specify even length sequence for nontrivial dataset!'
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab = ["(",")","{","}","[","]"]   
        self.token2idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.data = self.generate_data()
    
    def is_balanced(self, s):
        stack = []
        matching_bracket = {')': '(', '}': '{', ']': '['}
        for char in s:
            if char in '({[':
                stack.append(char)
            elif char in ')}]':
                if not stack or stack.pop() != matching_bracket[char]:
                    return 0  # Unbalanced
        return 1 if not stack else 0
    
    def generate_data(self):
        data = []
        brackets = "(){}[]"
        balanced_count = 0
        unbalanced_count = 0
        target_count = self.num_samples // 2 # around 50% balanced 

        while len(data) < self.num_samples:
            seq = ''.join(random.choice(brackets) for _ in range(self.seq_length))
            label = self.is_balanced(seq)
            if (label == 1 and balanced_count < target_count) or (label == 0 and unbalanced_count < target_count):     
                seq_tensor = torch.tensor([brackets.index(ch) for ch in seq], dtype=torch.float32)
                label_tensor = torch.tensor(label, dtype=torch.long)
                data.append((seq_tensor, label_tensor))
            
            if label == 1:
                balanced_count += 1
            else:
                unbalanced_count += 1
        
        return data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
        
        


class IsPalindromeDataset(Dataset):
    '''
    Return the dataset for two tasks related to Palindrome:
    1. IsPalindrome (invariant task): given a sequence, if it contains a palindrome substring with length=palindrome_length, returns label 1, otherwise 0
    2. IsPalindrome (equivariant task): same as above, except returning labels as a binary sequence indicating the locations of the palindrome substring
    Note that for the equivariant task, the generated string may contain palindrome with length less than the specified palindrome_length,
    e.g. palindrome_length = 4, generated [6., 8., 5., 8., 5.] is not a palindrome with length 4 (TODO: may modify this if we allow smaller palindromes)
    '''
    def __init__(self, num_samples=10000, seq_length=5, palindrome_length=3, seed=99, equivariant=False):
        set_seed(seed)
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.equivariant = equivariant
        self.palindrome_length = palindrome_length 
        if self.equivariant:
            self.data = self.generate_equiv_data()
        else:
            self.data = self.generate_inv_data()
    
    def generate_equiv_data(self):
        data = []
        for _ in range(self.num_samples):
            while True:
                # generate non-palindromes
                seq = [random.choice(string.digits) for _ in range(self.seq_length)]
                label = [0] * self.seq_length  # Binary label sequence
                
                # generate palindrome substrings
                if random.random() > 0.5:
                    # Insert a palindrome substring
                    start_idx = random.randint(0, self.seq_length - self.palindrome_length)
                    half = [random.choice(string.digits) for _ in range(self.palindrome_length // 2)]
                    if self.palindrome_length % 2 == 0:
                        palindrome = half + half[::-1]
                    else:
                        palindrome = half + [random.choice(string.digits)] + half[::-1]
                    seq[start_idx:start_idx + self.palindrome_length] = palindrome
                    label[start_idx:start_idx + self.palindrome_length] = [1] * self.palindrome_length
                
                # ensure non-palindromes truly don't contain a palindrome substring
                where_palindrome = [int(seq[i:i+self.palindrome_length] == seq[i:i+self.palindrome_length][::-1])
                    for i in range(self.seq_length - self.palindrome_length + 1)]

                has_palindrome = any(where_palindrome)
                if (not any(label)) and has_palindrome:
                    continue  # Regenerate sequence if a palindrome substring exists but isn't labeled
                elif any(label): #post processing check on the palindrome label (update for additional larger palindorme)
                    for idx, val in enumerate(where_palindrome):
                        if val ==1 and idx != start_idx:
                            label[idx:(idx + self.palindrome_length)] = [1] * self.palindrome_length
                    
                seq_tensor = torch.tensor([int(ch) for ch in seq], dtype=torch.float32)
                label_tensor = torch.tensor(label, dtype=torch.long)
                data.append((seq_tensor, label_tensor))
                break  # Exit loop once a valid sequence is generated
        
        return data
    
    def generate_inv_data(self):
        data = []
        for _ in range(self.num_samples):
            while True:
                # generate non-palindromes
                seq = [random.choice(string.digits) for _ in range(self.seq_length)]
                label = 0  
                
                # generate palindrome substrings
                if random.random() > 0.5:
                    # Insert a palindrome substring
                    start_idx = random.randint(0, self.seq_length - self.palindrome_length)
                    half = [random.choice(string.digits) for _ in range(self.palindrome_length // 2)]
                    if self.palindrome_length % 2 == 0:
                        palindrome = half + half[::-1]
                    else:
                        palindrome = half + [random.choice(string.digits)] + half[::-1]
                    seq[start_idx:start_idx + self.palindrome_length] = palindrome
                    label = 1
                
                # ensure non-palindromes truly don't contain a palindrome substring
                has_palindrome = any(
                    seq[i:i+self.palindrome_length] == seq[i:i+self.palindrome_length][::-1]
                    for i in range(self.seq_length - self.palindrome_length + 1)
                )
                if label == 0 and has_palindrome:
                    continue  # Regenerate sequence if a palindrome substring exists but isn't labeled
                
                seq_tensor = torch.tensor([int(ch) for ch in seq], dtype=torch.float32)
                label_tensor = torch.tensor(label, dtype=torch.long)
                data.append((seq_tensor, label_tensor))
                break  # Exit loop once a valid sequence is generated
        
        return data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
        

class IntersectDataset(Dataset):
    '''
    Return dataset for the set intersection task
    Input: (set1, set2) (concatenated as a input sequence)
    Labels: an integer for the invariant task, or a binary sequence for the equivariant task, where
    - Invariant task: Determine the size of the intersection of set1 and set2
    - Equivariant task: for each element in set1, determine if it is contained in set2, and vice versa 
    '''
    def __init__(self, num_samples=10000, seq_length=6, seed=42, equivariant=False, vocab_size=4):
        set_seed(seed)
        assert seq_length % 2 ==0, 'must pass in even sequence length!'
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.set_size = seq_length//2
        self.equiv = equivariant 
        self.vocab = list(string.ascii_lowercase)[:vocab_size]
        self.data = self.generate_data()
    
    def generate_data(self):
        data = []
        
        for _ in range(self.num_samples):
            set1 = set(random.sample(self.vocab, self.set_size))
            set2 = set(random.sample(self.vocab, self.set_size))            
            seq_list = list(set1) + list(set2)
            mid = len(seq_list) // 2
            target1 = [1 if ch in set2 else 0 for ch in seq_list[:mid]]
            target2 = [1 if ch in set1 else 0 for ch in seq_list[mid:]]
            target = target1 + target2
            seq_tensor = torch.tensor([ord(ch) for ch in seq_list], dtype=torch.float32)
            seq_tensor = seq_tensor - 97
            if self.equiv:
                target_tensor = torch.tensor(target, dtype=torch.long)
            else:
                target_tensor = min(target1.count(1), target2.count(1))
            
            data.append((seq_tensor, target_tensor))
        
        return data
    
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]




class MaxCyclicSumDataset(Dataset):
    '''
    Dataset for finding the unique contiguous (including cyclic wraparound) subsequence
    of a specified length (cyc_length) with the maximum sum.

    Each element in the sequence is an integer in the range [0, vocab_size - 1].

    Example:
    sequence = [7, 2, 4, 1, 9, 8], cyc_length=3
    label =    [1, 0, 0, 0, 1, 1]  # because [7, 9, 8] is the unique max-sum cyclic subarray
    '''
    def __init__(self, num_samples=10000, seq_length=6, cyc_length=3, vocab_size=10, seed=42):
        random.seed(seed)
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.cyc_length = cyc_length
        self.vocab_size = vocab_size
        self.data = self.generate_data()

    def generate_data(self):
        data = []
        while len(data) < self.num_samples:
            seq = [random.randint(0, self.vocab_size - 1) for _ in range(self.seq_length)]

            # Create a doubled version for wraparound
            doubled_seq = seq + seq[:self.cyc_length - 1]
            window_sums = []
            for i in range(self.seq_length):
                window_sums.append((sum(doubled_seq[i:i + self.cyc_length]), i))

            # Sort by descending sum
            window_sums.sort(reverse=True)
            max_sum, max_start = window_sums[0]

            # Check uniqueness
            if len(window_sums) > 1 and window_sums[0][0] == window_sums[1][0]:
                continue  # Not unique max sum → regenerate

            # Unique max found
            label = [0] * self.seq_length
            for i in range(self.cyc_length):
                label[(max_start + i) % self.seq_length] = 1

            seq_tensor = torch.tensor(seq, dtype=torch.long)
            label_tensor = torch.tensor(label, dtype=torch.long)
            data.append((seq_tensor, label_tensor))

        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]




# Example usage
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='palindrome', choices=['palindrome','balance','intersect', "cyclicsum"])

    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--seq_length', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--palindrome_length', type=int, default=2)
    parser.add_argument('--cyc_length', type=int, default=3)
    parser.add_argument('--equiv', action='store_true', help='use equivariant labels (i.e. same size as seq_length)')
    parser.add_argument('--vocab_size', type=int, default=4, help='vocab size for the set intersection (TODO: extend this for all other tasks)')
    parser.add_argument('--seed', type=int, default=9)

    args = parser.parse_args()

    if args.task == 'palindrome':
        dataset = IsPalindromeDataset(num_samples=args.num_samples, seq_length=args.seq_length, 
                                    palindrome_length=args.palindrome_length, seed=args.seed,
                                    equivariant=args.equiv)
    elif args.task == 'balance':
        dataset = IsBalancedParenthesisDataset(num_samples=args.num_samples, 
                                               seq_length=args.seq_length, seed=args.seed)
    elif args.task == 'intersect':
        dataset = IntersectDataset(num_samples=args.num_samples, 
                                   seq_length=args.seq_length, seed=args.seed,
                                   equivariant=args.equiv, vocab_size=args.vocab_size)
    elif args.task == 'cyclicsum':
        dataset = MaxCyclicSumDataset(num_samples=args.num_samples, 
                                   seq_length=args.seq_length, seed=args.seed,
                                   cyc_length=args.cyc_length, vocab_size = args.vocab_size)
    
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    for batch in dataloader:
        sequences, labels = batch
        print("Sequences:", sequences)
        print("Labels:", labels)
        break  # Print only first batch

if __name__ == "__main__":
    main()
