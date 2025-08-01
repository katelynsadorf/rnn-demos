##### Imports #####
from tqdm import tqdm 
import numpy as np

##### Data #####
train_X = ['h', 'e', 'l', 'l', 'o', ' ']
train_Y = ['e', 'l', 'l', 'o', ' ', 'w']

test_X = ['w', 'o', 'r', 'l']
test_Y = ['o', 'r', 'l', 'd']

vocab = sorted(list(set(train_X + train_Y + test_X +test_Y)))
vocab_size = len(vocab)

char_to_idx = {ch : i for i, ch in enumerate(vocab)}
idx_to_char = {i : ch for ch, i in enumerate(vocab)}

train_X_idx = [char_to_idx[ch] for ch in train_X]
train_Y_idx = [char_to_idx[ch] for ch in train_Y]

