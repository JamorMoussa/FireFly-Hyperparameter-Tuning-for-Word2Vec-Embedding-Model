import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.model_selection import train_test_split

# Sample text data
text = "We are learning word embeddings using the Continuous Bag of Words model."

# Preprocess text
def preprocess(text):
    text = text.lower().split()
    word_counts = Counter(text)
    vocabulary = {word: i for i, word in enumerate(word_counts)}
    return text, vocabulary

text, vocabulary = preprocess(text)
vocab_size = len(vocabulary)
word_to_idx = {word: i for i, word in enumerate(vocabulary)}
idx_to_word = {i: word for i, word in enumerate(vocabulary)}

# Create context-target pairs for CBOW
def create_context_target_pairs(text, window_size):
    context_target_pairs = []
    for i in range(window_size, len(text) - window_size):
        context = [text[i - j - 1] for j in range(window_size)] + [text[i + j + 1] for j in range(window_size)]
        target = text[i]
        context_target_pairs.append((context, target))
    return context_target_pairs

window_size = 2  # Example window size
context_target_pairs = create_context_target_pairs(text, window_size)

# Define CBOW model
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim * window_size , vocab_size)
    
    def forward(self, context):
        embeds = self.embeddings(context).view(1, -1)
        out = self.linear(embeds)
        return out

model = CBOW(vocab_size, 10)
print(model)

# Firefly Algorithm parameters
num_fireflies = 10
num_iterations = 30
alpha = 0.2
beta0 = 1.0
gamma = 1.0

# Hyperparameter search space
param_bounds = {
    "learning_rate": (1e-4, 1e-2),
    "embedding_dim": (50, 300),
    "window_size": (1, 5),
}

# Initialize fireflies
fireflies = np.random.uniform(low=[param_bounds[key][0] for key in param_bounds], 
                              high=[param_bounds[key][1] for key in param_bounds], 
                              size=(num_fireflies, len(param_bounds)))

def evaluate_firefly(params):
    learning_rate = params[0]
    embedding_dim = int(params[1])
    window_size = int(params[2])
    
    context_target_pairs = create_context_target_pairs(text, window_size)
    
    model = CBOW(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Train the model
    def train(model, context_target_pairs, optimizer, criterion):
        model.train()
        total_loss = 0
        for context, target in context_target_pairs:
            context_idxs = torch.tensor([word_to_idx[w] for w in context], dtype=torch.long)
            target_idx = torch.tensor([word_to_idx[target]], dtype=torch.long)
            
            optimizer.zero_grad()
            output = model(context_idxs)
            loss = criterion(output, target_idx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(context_target_pairs)
    
    # Evaluate the model
    train_loss = train(model, context_target_pairs, optimizer, criterion)
    return train_loss

# Evaluate initial fireflies
fitness = np.array([evaluate_firefly(firefly) for firefly in fireflies])

# Firefly Algorithm optimization loop
for iteration in range(num_iterations):
    for i in range(num_fireflies):
        for j in range(num_fireflies):
            if fitness[i] > fitness[j]:
                r = np.linalg.norm(fireflies[i] - fireflies[j])
                beta = beta0 * np.exp(-gamma * r ** 2)
                fireflies[i] = fireflies[i] + beta * (fireflies[j] - fireflies[i]) + alpha * (np.random.rand(len(param_bounds)) - 0.5)
                fireflies[i] = np.clip(fireflies[i], [param_bounds[key][0] for key in param_bounds], [param_bounds[key][1] for key in param_bounds])
                fitness[i] = evaluate_firefly(fireflies[i])
    print(f"Iteration {iteration + 1}/{num_iterations}, Best Loss: {fitness.min()}")

# Best hyperparameters
best_index = np.argmin(fitness)
best_params = fireflies[best_index]
best_learning_rate = best_params[0]
best_embedding_dim = int(best_params[1])
best_window_size = int(best_params[2])

print(f"Best Learning Rate: {best_learning_rate}")
print(f"Best Embedding Dimension: {best_embedding_dim}")
print(f"Best Window Size: {best_window_size}")
