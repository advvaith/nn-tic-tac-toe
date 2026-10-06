import torch
from torch import nn, relu, sigmoid


if torch.backends.mps.is_available():
    device = torch.device('mps')

    print(f"using {device}")
    
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.dl1 = nn.Linear(9, 32)
        self.dl2 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, 9)
        
    def forward(self, x):
        x = self.dl1(x)
        x = relu(x)
        
        x = self.dl2(x)
        x = relu(x)
        
        x = self.output_layer(x)
        x = sigmoid(x)
        
        return x