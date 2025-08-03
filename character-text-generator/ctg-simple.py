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

##### Recurrent Neural Network Class #####
class RNN:
    def __init__(self, input_size, hidden_size, output_size, num_epochs, learning_rate):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Network
        self.w1 = initWeights(input_size, hidden_size)
        self.w2 = initWeights(input_size, hidden_size)
        self.w3 = initWeights(hidden_size, output_size)

        self.b2 = np.zeros((1, hidden_size))
        self.b3 = np.zeros((1, output_size))

# Forward Propogation
def forward(self, inputs):
    self.hidden_states = [np.zeros_like(self.b2)]

    for input in inputs:
        layer1_output = np.dot(input, self.w1)
        layer2_output = np.dot(self.hidden_states[-1], self.w2) + self.b2

        self.hidden_states += [tanh(layer1_output +layer2_output)]

    return np.dot(self.hidden_states[-1, self.w3] + self.b3)

# Backward Propogation
def backward(self, error, inputs):
    d_b3 = error
    d_w3 = np.dot(self.hidden_states[-1].T, error)

    d_b2 = np.zeros_like(self.b2)
    d_w2 = np.zeros_like(self.w2)
    d_w1 = np.zeros_like(self.w1)

    d_hidden_state = np.dot(error, self.w3.T)
    for q in reversed(range(len(inputs))):
        d_hidden_state *= tanh(self.hidden_states[q + 1], derivative = True)

        d_b2 += d_hidden_state

        d_w2 += np.dot(self.hidden_states[q].T, d_hidden_state)

        d_w1 += np.dot(inputs[q].T, d_hidden_state)

        d_hidden_state = np.dot(d_hidden_state, self.w2)

    for d_ in (d_b3, d_w3, d_b2, d_w2, d_w1):
        np.clip(d_, -1, 1, out = d_)

    self.b3 += self.learning_rate * d_b3
    self.w3 += self.learning_rate * d_w3
    self.b2 += self.learning_rate * d_b2
    self.w2 += self.learning_rate * d_w2
    self.w1 += self.learning_rate * d_w1
