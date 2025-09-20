import torch
from torch.utils.data import Dataset, DataLoader
import random
import string
import argparse
from collections import Counter, defaultdict
from itertools import combinations


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

                # has_palindrome = any(where_palindrome)
                # if (not any(label)) and has_palindrome:
                #     continue  # Regenerate sequence if a palindrome substring exists but isn't labeled
                # elif any(label): #post processing check on the palindrome label (update for additional larger palindorme)
                #     for idx, val in enumerate(where_palindrome):
                #         if val ==1 and idx != start_idx:
                #             label[idx:(idx + self.palindrome_length)] = [1] * self.palindrome_length

                # after generating where_palindrome
                has_palindrome = any(where_palindrome)
                
                if not any(label) and has_palindrome:
                    continue  # Regenerate sequence if unexpected palindrome
                
                elif any(label):
                    # For all detected palindromes, update label accordingly
                    for idx, val in enumerate(where_palindrome):
                        if val == 1:
                            for i in range(self.palindrome_length):
                                if (idx + i) < self.seq_length:
                                    label[idx + i] = 1

                    
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
    def __init__(self, num_samples=10000, seq_length=8, seed=42, equivariant=False, vocab_size=4, thresh = 3):
        set_seed(seed)
        assert seq_length % 2 ==0, 'must pass in even sequence length!'
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.set_size = seq_length//2
        self.equiv = equivariant 
        self.vocab = list(string.ascii_lowercase)[:vocab_size]
        self.thresh = thresh
        self.data = self.generate_data()
        
    
    def generate_data(self):
        data = []
        
        for _ in range(self.num_samples):
            set1_list  = [random.choice(self.vocab) for _ in range(self.set_size)]
            set2_list = [random.choice(self.vocab) for _ in range(self.set_size)]
            seq_list  = set1_list + set2_list
            mid = len(seq_list) // 2
            set1_set = set(set1_list)
            set2_set = set(set2_list)
            target1 = [1 if ch in set2_set else 0 for ch in set1_list]
            target2 = [1 if ch in set1_set else 0 for ch in set2_list]
            target = target1 + target2
            seq_tensor = torch.tensor([ord(ch) for ch in seq_list], dtype=torch.float32)
            seq_tensor = seq_tensor - 97
            if self.equiv:
                target_tensor = torch.tensor(target, dtype=torch.long)
            else:
                target_tensor = len(set1_set & set2_set)
                #target_tensor = 1 if len(set1 & set2) >= self.thresh else 0
            
            data.append((seq_tensor, target_tensor))
        
        return data
    
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
        
        


class Vandermonde(Dataset):
    '''
    Dataset that checks the sign of the Vandermonde determinant
    '''
    def __init__(self, num_samples=10000, seq_length=8, seed=42, vocab_size=15):
        set_seed(seed)
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.data = self.generate_data()
        
    def generate_data(self):
        data = []
        for _ in range(self.num_samples):
            # Sample seq_length distinct integers from 1 to vocab_size
            seq = random.sample(range(1, self.vocab_size + 1), self.seq_length)
            
            # Count inversions: number of (i, j) pairs with i < j and seq[i] > seq[j]
            inversions = 0
            for i in range(self.seq_length):
                for j in range(i + 1, self.seq_length):
                    if seq[i] > seq[j]:
                        inversions += 1

            # Label: 1 if even number of inversions (positive sign), -1 if odd
            label = 1 if inversions % 2 == 0 else 0

            seq_tensor = torch.tensor(seq, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.long)
            data.append((seq_tensor, label_tensor))
        return data

    def __len__(self):
        return len(self.data)

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
    def __init__(self, num_samples=10000, seq_length=6, cyc_length=3, vocab_size=10, seed=42, inv = False):
        random.seed(seed)
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.cyc_length = cyc_length
        self.vocab_size = vocab_size
        self.inv = inv
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
            
            if self.inv:
                seq_tensor = torch.tensor(seq, dtype=torch.long)
                label_tensor = torch.tensor([max_sum], dtype=torch.float32)
            
            else:
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

class LongestPalindromeDataset(Dataset):
    """
    Given a string of lowercase letters, label = length of the longest palindrome
    that can be formed by permuting those letters.
    E.g. 'abccccdd' → 7  (dccaccd)

    Symmetry: invariant to S_n, all permutations of the input
    """
    def __init__(self, num_samples=10000, seq_length=10, seed=42,
                 vocab_size=20, thresh = 4):
        set_seed(seed)
        self.num_samples = num_samples
        self.thresh = thresh
        self.seq_length = seq_length
        # default to full lowercase alphabet
        assert 2 <= vocab_size <= 26, "Vocab size must be in reasonable range"
        self.vocab = list("abcdefghijklmnopqrstuvwxyz")[:vocab_size]
        self.token2idx = {ch: i for i, ch in enumerate(self.vocab)}
        pool = self._generate(num_samples*10)
        self.data = self._rebalance(pool)
        

    def _compute_label(self, seq):
        cnt = Counter(seq)
        # sum of even parts
        pair_sum = sum((c // 2) * 2 for c in cnt.values())
        # if any odd leftover, we can put one in the middle
        #return 1 if (pair_sum + (1 if any(c % 2 for c in cnt.values()) else 0)) >= self.thresh else 0
        return pair_sum + (1 if any(c % 2 for c in cnt.values()) else 0)
    def _generate(self, num_samples):
        data = []
        for _ in range(num_samples):
            # sample a random string
            seq = [random.choice(self.vocab) for _ in range(self.seq_length)]
            lab = self._compute_label(seq)
            # map chars → indices
            seq_tensor = torch.tensor([self.token2idx[ch] for ch in seq],
                                      dtype=torch.float32)
            label_tensor = torch.tensor(lab, dtype=torch.long)
            data.append((seq_tensor, label_tensor))
        return data

    def _rebalance(self, pool):
        # bucket by label
        buckets = defaultdict(list)
        for seq, lab in pool:
            buckets[int(lab.item())].append((seq, lab))

        labels = list(buckets.keys())
        num_classes = len(labels)
        per_class = self.num_samples // num_classes

        balanced = []
        for lab in labels:
            samples = buckets[lab]
            if len(samples) >= per_class:
                chosen = random.sample(samples, per_class)
            else:
                # if too few, allow repeats
                chosen = samples + random.choices(samples, k=per_class - len(samples))
            balanced.extend(chosen)

        # if there's any slack (due to integer division), fill from the pool
        if len(balanced) < self.num_samples:
            extra = random.sample(pool, self.num_samples - len(balanced))
            balanced.extend(extra)
        # or trim if we overshot
        random.shuffle(balanced)
        return balanced[: self.num_samples]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

class DetectCapitalDataset(Dataset):
    """
    LeetCode “Detect Capital”:
    Label 1 if word is all caps, all lowercase,
    or only first letter capitalized; else 0.

    e.g. USA = 1
         FlooR = 0
         tremendous = 1
         Sun = 1

    Invariant task: symmetry group is S_{n-1}, you can freely permute all except first token.
    """
    def __init__(self, num_samples=10000, word_length=6, seed=42):
        # recommended word length of 6, so that there are a decent number
        # of ULLLLL examples

        set_seed(seed)
        self.num_samples = num_samples
        self.word_length = word_length
        lows = list(string.ascii_lowercase)
        ups  = [c.upper() for c in lows]
        # map a–z → 0–25, A–Z → 26–51
        self.char2idx = {c:i for i,c in enumerate(lows)}
        self.char2idx.update({C:i+26 for i,C in enumerate(ups)})
        assert 52**word_length > num_samples * 2, "Not enough data to generate"
        self.data = self._generate()

    def _is_correct(self, w):
        return w.isupper() or w.islower() or (w[0].isupper() and w[1:].islower())

    def _generate(self):
        data, t_true, t_false = [], 0, 0
        half = self.num_samples // 2
        all_chars = list(self.char2idx.keys())
        while len(data) < self.num_samples:
            w = ''.join(random.choice(all_chars) for _ in range(self.word_length))
            lbl = int(self._is_correct(w))
            if (lbl == 1 and t_true  < half) or (lbl == 0 and t_false < half):
                seq = torch.tensor([self.char2idx[ch] for ch in w],
                                   dtype=torch.float32)
                lab = torch.tensor(lbl, dtype=torch.long)
                data.append((seq, lab))
                if lbl: t_true  += 1
                else:   t_false += 1
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]



class SetGameDataset(Dataset):
    '''
    Generate a dataset for the card game SET.
    Each sample consists of:
    - input: 10 cards, each with 4 attributes (each attribute ∈ {0,1,2}), flattened into a vector of size 40
    - label: length-10 binary vector; label[i]=1 if card i is in some SET among the 10 cards, else 0
    '''
    def __init__(self, seq_length = 10, num_samples=30, seed=42):
        random.seed(seed)
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.data = self.generate_data()
    
    def is_set(self, card1, card2, card3):
        # A set if for each attribute: all same or all different
        for a, b, c in zip(card1, card2, card3):
            if (a == b == c) or (len({a, b, c}) == 3):
                continue
            else:
                return False
        return True
    def card_to_int(self, card):
        # Interpret 4-tuple (a,b,c,d) as a base-3 number
        # a, b, c, d = card
        # return a * (3**3) + b * (3**2) + c * (3**1) + d * (3**0)
        a, b, c = card
        return a * (3**2) + b * (3**1) + c * (3**0) 

    def generate_data(self):
        #all_cards = [(a,b,c,d) for a in range(3) for b in range(3) for c in range(3) for d in range(3)]
        all_cards = [(a,b,c) for a in range(3) for b in range(3) for c in range(3)]
        data = []
        
        for _ in range(self.num_samples):
           
            cards = random.sample(all_cards, self.seq_length)
            labels = [0] * self.seq_length  # Initialize labels

            # Check all triplets
            for i, j, k in combinations(range(self.seq_length), 3):
                if self.is_set(cards[i], cards[j], cards[k]):
                    labels[i] = labels[j] = labels[k] = 1

            # Flatten the cards
            int_cards = [self.card_to_int(card) for card in cards]
            seq_tensor = torch.tensor(int_cards, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.long)
            data.append((seq_tensor, label_tensor))
        
        return data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

class SETIntersect(Dataset):
    '''
    SETIntersect Dataset:
    - Two sequences of random SET cards (length seq_length/2 each)
    - Input: sequence1 + sequence2 (each card hashed to int)
    - Label:
        - For each card in sequence1: does it form a SET with two distinct cards from sequence2?
        - For each card in sequence2: does it form a SET with two distinct cards from sequence1?
    '''
    def __init__(self, num_samples=10, seq_length=18, seed=42):
        assert seq_length % 2 == 0, 'seq_length must be even!'
        random.seed(seed)
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.subseq_length = seq_length // 2
        self.data = self.generate_data()
    
    def is_set(self, card1, card2, card3):
        for a, b, c in zip(card1, card2, card3):
            if (a == b == c) or (len({a, b, c}) == 3):
                continue
            else:
                return False
        return True

    def card_to_int(self, card):
        # a, b, c, d = card
        # return a * (3**3) + b * (3**2) + c * (3**1) + d * (3**0)
        a, b, c = card
        return a * (3**2) + b * (3**1) + c * (3**0) 

    def generate_data(self):
        # all_cards = [(a,b,c,d) for a in range(3) for b in range(3) for c in range(3) for d in range(3)]
        all_cards = [(a,b,c) for a in range(3) for b in range(3) for c in range(3)]
        data = []
        
        for _ in range(self.num_samples):
            # Sample sequence1 and sequence2 independently
            cards_seq1 = random.sample(all_cards, self.subseq_length)
            cards_seq2 = random.sample(all_cards, self.subseq_length)

            # Final concatenated sequence
            full_sequence = cards_seq1 + cards_seq2

            labels_seq1 = [0] * self.subseq_length
            labels_seq2 = [0] * self.subseq_length

            # Labeling for sequence1
            for idx1, card1 in enumerate(cards_seq1):
                for j, k in combinations(range(self.subseq_length), 2):
                    card2 = cards_seq2[j]
                    card3 = cards_seq2[k]
                    if self.is_set(card1, card2, card3):
                        labels_seq1[idx1] = 1
                        break  # found a SET for this card

            # Labeling for sequence2
            for idx2, card2 in enumerate(cards_seq2):
                for j, k in combinations(range(self.subseq_length), 2):
                    card1 = cards_seq1[j]
                    card3 = cards_seq1[k]
                    if self.is_set(card2, card1, card3):
                        labels_seq2[idx2] = 1
                        break  # found a SET for this card

            # Merge labels
            labels = labels_seq1 + labels_seq2

            # Hash the cards
            int_cards = [self.card_to_int(card) for card in full_sequence]

            seq_tensor = torch.tensor(int_cards, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.long)
            data.append((seq_tensor, label_tensor))
        
        return data

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]



# Example usage
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='palindrome', choices=['palindrome','balance','intersect', "cyclicsum", "longestpal", "detectcapital", "set", "setintersect", 'vandermonde'])

    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--seq_length', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--palindrome_length', type=int, default=2)
    parser.add_argument('--cyc_length', type=int, default=3)
    parser.add_argument('--equiv', action='store_true', help='use equivariant labels (i.e. same size as seq_length)')
    parser.add_argument('--vocab_size', type=int, default=4, help='vocab size for the set intersection (TODO: extend this for all other tasks)')
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--thresh', type=int, default = 4)

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
                                   equivariant=args.equiv, vocab_size=args.vocab_size, thresh = args.thresh)
    elif args.task == 'cyclicsum':
        dataset = MaxCyclicSumDataset(num_samples=args.num_samples, 
                                   seq_length=args.seq_length, seed=args.seed,
                                   cyc_length=args.cyc_length, vocab_size = args.vocab_size)
    elif args.task == 'vandermonde':
        dataset = Vandermonde(num_samples=args.num_samples, 
                                   seq_length=args.seq_length, seed=args.seed,
                                   vocab_size = args.vocab_size)
    elif args.task == 'longestpal':
        dataset = LongestPalindromeDataset(
            num_samples=args.num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            thresh = args.thresh
        )
    elif args.task == 'detectcapital':
        dataset = DetectCapitalDataset(
            num_samples=args.num_samples,
            word_length=args.seq_length,
            seed=args.seed
        )
    elif args.task == 'set':
        dataset = SetGameDataset(num_samples = args.num_samples,seed=args.seed, seq_length = args.seq_length)
    elif args.task == 'setintersect':
        dataset = SETIntersect(num_samples = args.num_samples, seed = args.seed, seq_length = args.seq_length)
    else:
        raise ValueError("Invalid task")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    for batch in dataloader:
        sequences, labels = batch
        print("Sequences:", sequences)
        print("Labels:", labels)
        break  # Print only first batch

    #dist = Counter(int(label.item()) for _, label in dataset)
    #print("Label distribution:", dist)

if __name__ == "__main__":
    main()