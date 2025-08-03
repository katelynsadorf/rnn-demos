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

##### Helper Functions #####

def one_hot(index, vocab_size):
    vec = np.zeros((vocab_size,))
    vec[index] = 1
    return vec

# Xavier Normalized Initialization
def initWeights(input_size, output_size):
    return np.random.uniform(-1, 1, (input_size, output_size)) * np.sqrt(6 / (input_size + output_size))

##### Activation Functions ######
def tanh(input, derivative = False):
    if derivative:
        return 1 - (input ** 2)
    
    return np.tanh(input)

def softmax(input):
    return np.exp(input) / np.sum(np.exp(input))