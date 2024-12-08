import torch.nn as nn

class simpleMLP(nn.Module):
    def __init__(self, input_size):
        super(simpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.simpleMLP(x)
