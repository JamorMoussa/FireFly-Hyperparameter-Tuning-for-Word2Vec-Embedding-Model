import torch
import torch.nn as nn
import torchfly.optim as optim

# Create a linear regression model
model = nn.Sequential(
    nn.Linear(3, 1)
)

# Define the loss function
loss_fn = nn.MSELoss()

# Create an instance of FireFlyOptimizer
opt = optim.FireFlyOptimizer(model, loss_fn)

# Generate inputs and targets
inputs = torch.randn(100, 3)
targets = torch.mm(inputs, torch.Tensor([[1, 2, 3]]).t())


for x, y in zip(inputs, targets):
    opt.step(x, y)

print(model[0].weight)