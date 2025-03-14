import torch
from torch.utils.data import Dataset, DataLoader
import random
import string
import argparse

def set_seed(seed=99):
    random.seed(seed)
    torch.manual_seed(seed)

class IsPalindromeDataset(Dataset):
    '''
    Return the dataset for two tasks related to Palindrome:
    1. IsPalindrome (invariant task): given a sequence, if it contains a palindrome substring with length=palindrome_length, returns label 1, otherwise 0
    2. IsPalindrome (equivariant task): same as above, except returning labels as a binary sequence indicating the locations of the palindrome substring
    Note that for the equivariant task, the generated string may contain palindrome with length greater that the specified palindrome_length,
    e.g. palindrome_length = 2, generated [4., 4., 4, 0., 0.]
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

# Example usage
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--seq_length', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--palindrome_length', type=int, default=2)
    parser.add_argument('--equiv', action='store_true', help='use equivariant labels (i.e. same size as seq_length)')

    parser.add_argument('--seed', type=int, default=9)

    args = parser.parse_args()


    dataset = IsPalindromeDataset(num_samples=args.num_samples, seq_length=args.seq_length, 
                                  palindrome_length=args.palindrome_length, seed=args.seed,
                                  equivariant=args.equiv)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    for batch in dataloader:
        sequences, labels = batch
        print("Sequences:", sequences)
        print("Labels:", labels)
        break  # Print only first batch

if __name__ == "__main__":
    main()
