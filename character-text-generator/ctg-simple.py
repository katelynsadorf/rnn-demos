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
idx_to_char = {i : ch for i, ch in enumerate(vocab)}

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
        return 1 - np.tanh(input) ** 2
    
    return np.tanh(input)

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

##### Recurrent Neural Network Class #####
class RNN:
    def __init__(self, input_size, hidden_size, output_size, num_epochs, learning_rate):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Network
        self.w1 = initWeights(input_size, hidden_size)
        self.w2 = initWeights(hidden_size, hidden_size)
        self.w3 = initWeights(hidden_size, output_size)

        self.b2 = np.zeros((1, hidden_size))
        self.b3 = np.zeros((1, output_size))

    # Forward Propogation
    def forward(self, inputs):
        self.hidden_states = [np.zeros_like(self.b2)]
        self.pre_activations = []

        for input in inputs:
            layer1_output = np.dot(input, self.w1)
            layer2_output = np.dot(self.hidden_states[-1], self.w2) + self.b2

            pre_activation = layer1_output + layer2_output
            self.pre_activations.append(pre_activation)

            new_hidden = tanh(pre_activation)
            self.hidden_states.append(new_hidden.reshape(1, -1))


        return np.dot(self.hidden_states[-1], self.w3) + self.b3

    # Backward Propogation
    def backward(self, error, inputs):
        d_b3 = error
        d_w3 = np.dot(self.hidden_states[-1].T, error)

        d_b2 = np.zeros_like(self.b2)
        d_w2 = np.zeros_like(self.w2)
        d_w1 = np.zeros_like(self.w1)

        d_hidden_state = np.dot(error, self.w3.T)
        for q in reversed(range(len(inputs))):
            d_hidden_state *= tanh(self.pre_activations[q], derivative = True)

            d_b2 += d_hidden_state

            d_w2 += np.dot(self.hidden_states[q].T, d_hidden_state)

            d_w1 += np.dot(inputs[q].T, d_hidden_state)

            d_hidden_state = np.dot(d_hidden_state, self.w2.T)

        for d_ in (d_b3, d_w3, d_b2, d_w2, d_w1):
            np.clip(d_, -1, 1, out = d_)

        self.b3 += self.learning_rate * d_b3
        self.w3 += self.learning_rate * d_w3
        self.b2 += self.learning_rate * d_b2
        self.w2 += self.learning_rate * d_w2
        self.w1 -= self.learning_rate * d_w1

    # Train
    def train(self, inputs, labels):
        for _ in tqdm(range(self.num_epochs)):
            for x_char, y_char in zip(inputs, labels):
                x_idx = char_to_idx[x_char]
                y_idx = char_to_idx[y_char]

                x_vec = one_hot(x_idx, vocab_size).reshape(1, -1)
                y_true = y_idx

                y_pred = self.forward([x_vec])

                probs = softmax(y_pred).flatten()
                loss = -np.log(probs[y_idx] + 1e-8)
                epoch_loss += loss

                error = probs.copy()
                error[y_idx] -= 1
                error = error.reshape(1, -1)
                self.backward(error, [x_vec])

                self.backward(error, [x_vec])

            self.loss_history.append(epoch_loss / len(inputs))

    # Test Accuracy
    def test_accuracy(self, inputs, labels):
        correct = 0
        for x_char, y_char in zip(inputs, labels):
            x_idx = char_to_idx[x_char]
            y_idx = char_to_idx[y_char]
            x_vec = one_hot(x_idx, vocab_size).reshape(1, -1)
            y_pred = self.forward([x_vec])
            pred_idx = int(np.argmax(y_pred))
            if pred_idx == y_idx:
                correct += 1
        return correct / len(inputs)

# Single Run
def single_run(hidden_size, learning_rate, epochs, seed):
    """Run a single training experiment"""
    rnn = RNN(input_size=vocab_size, hidden_size=hidden_size, 
              output_size=vocab_size, num_epochs=epochs, 
              learning_rate=learning_rate, seed=seed)
    
    rnn.train(train_X, train_Y)
    accuracy = rnn.test_accuracy(test_X, test_Y)
    final_loss = rnn.loss_history[-1]
    
    return {
        'accuracy': accuracy,
        'final_loss': final_loss,
        'converged': final_loss < rnn.loss_history[0] * 0.5,
        'seed': seed
    }

# Multiple Runs
def multiple_runs_experiment(hidden_size, learning_rate, epochs, num_runs=10):
    """Run multiple experiments and analyze statistics"""
    print(f"Running {num_runs} experiments...")
    print(f"Config: hidden_size={hidden_size}, lr={learning_rate}, epochs={epochs}")
    
    results = []
    seeds = range(42, 42 + num_runs)  # Different seeds for each run
    
    for seed in tqdm(seeds, desc="Running experiments"):
        result = single_run(hidden_size, learning_rate, epochs, seed)
        results.append(result)
    
    return results

# Fine-tune Analysis
def analyze_multiple_runs(results):
    """Analyze statistics from multiple runs"""
    accuracies = [r['accuracy'] for r in results]
    losses = [r['final_loss'] for r in results]
    convergence_rate = sum(r['converged'] for r in results) / len(results)
    
    stats = {
        'mean_accuracy': statistics.mean(accuracies),
        'std_accuracy': statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
        'min_accuracy': min(accuracies),
        'max_accuracy': max(accuracies),
        'median_accuracy': statistics.median(accuracies),
        'mean_loss': statistics.mean(losses),
        'std_loss': statistics.stdev(losses) if len(losses) > 1 else 0,
        'convergence_rate': convergence_rate,
        'num_runs': len(results)
    }
    
    return stats

def compare_configurations():
    """Compare different configurations with multiple runs"""
    configurations = [
        # (hidden_size, learning_rate, epochs)
        (16, 0.1, 1000),   # Small network, high LR
        (32, 0.05, 1000),  # Medium network, medium LR  
        (32, 0.1, 500),    # Medium network, high LR, fewer epochs
        (64, 0.01, 1500),  # Large network, low LR, more epochs
    ]
    
    print("=== CONFIGURATION COMPARISON (Multiple Runs) ===")
    all_configs = []
    
    for i, (hidden_size, lr, epochs) in enumerate(configurations):
        print(f"\n--- Configuration {i+1} ---")
        
        # Run multiple experiments for this configuration
        results = multiple_runs_experiment(hidden_size, lr, epochs, num_runs=5)
        stats = analyze_multiple_runs(results)
        
        # Store for comparison
        config_info = {
            'config_id': i+1,
            'hidden_size': hidden_size,
            'learning_rate': lr,
            'epochs': epochs,
            **stats
        }
        all_configs.append(config_info)
        
        # Print results
        print(f"Mean Accuracy: {stats['mean_accuracy']:.2%} ± {stats['std_accuracy']:.2%}")
        print(f"Range: {stats['min_accuracy']:.2%} - {stats['max_accuracy']:.2%}")
        print(f"Convergence Rate: {stats['convergence_rate']:.0%}")
        print(f"Mean Loss: {stats['mean_loss']:.4f} ± {stats['std_loss']:.4f}")
    
    # Find best configuration
    best_config = max(all_configs, key=lambda x: x['mean_accuracy'])
    most_reliable = min(all_configs, key=lambda x: x['std_accuracy'])
    
    print(f"\n=== SUMMARY ===")
    print(f"Best Mean Performance: Config {best_config['config_id']} - {best_config['mean_accuracy']:.2%}")
    print(f"Most Reliable (lowest std): Config {most_reliable['config_id']} - {most_reliable['std_accuracy']:.3f} std")
    
    return all_configs

def variance_analysis():
    """Analyze how much variance comes from random initialization"""
    print("\n=== VARIANCE ANALYSIS ===")
    
    # Fixed configuration, multiple seeds
    results_10_runs = multiple_runs_experiment(32, 0.1, 1000, num_runs=10)
    stats_10 = analyze_multiple_runs(results_10_runs)
    
    results_20_runs = multiple_runs_experiment(32, 0.1, 1000, num_runs=20)
    stats_20 = analyze_multiple_runs(results_20_runs)
    
    print(f"10 runs - Mean: {stats_10['mean_accuracy']:.2%} ± {stats_10['std_accuracy']:.3f}")
    print(f"20 runs - Mean: {stats_20['mean_accuracy']:.2%} ± {stats_20['std_accuracy']:.3f}")
    
    # Confidence intervals (assuming normal distribution)
    margin_10 = 1.96 * stats_10['std_accuracy'] / np.sqrt(10)  # 95% CI
    margin_20 = 1.96 * stats_20['std_accuracy'] / np.sqrt(20)
    
    print(f"95% Confidence Intervals:")
    print(f"10 runs: {stats_10['mean_accuracy']:.2%} ± {margin_10:.3f}")
    print(f"20 runs: {stats_20['mean_accuracy']:.2%} ± {margin_20:.3f}")


# Running
if __name__ == "__main__":
    # Run the comprehensive analysis
    print("Multiple Runs Fine-tuning Analysis")
    print("="*40)
    
    # Show why multiple runs matter
    print("\n1. Single vs Multiple Runs Comparison:")
    single_result = single_run(32, 0.1, 1000, seed=42)
    print(f"Single run (seed=42): {single_result['accuracy']:.2%}")
    
    multiple_results = multiple_runs_experiment(32, 0.1, 1000, num_runs=5)
    multiple_stats = analyze_multiple_runs(multiple_results)
    print(f"5 runs average: {multiple_stats['mean_accuracy']:.2%} ± {multiple_stats['std_accuracy']:.3f}")
    print(f"Range: {multiple_stats['min_accuracy']:.2%} - {multiple_stats['max_accuracy']:.2%}")
    
    # Compare configurations properly
    print("\n2. Configuration Comparison:")
    all_configs = compare_configurations()
    
    # Analyze variance
    print("\n3. Variance Analysis:")
    variance_analysis()
    
    # Give practical advice
    practical_recommendations()