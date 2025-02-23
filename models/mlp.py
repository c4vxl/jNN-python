import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_proj, out_proj, num_hidden, hidden_size, bias = True):
        super().__init__()

        self.in_proj = nn.Linear(in_proj, hidden_size, bias)
        self.out_proj = nn.Linear(hidden_size, out_proj, bias)
        self.hidden = nn.ModuleList([ nn.Linear(hidden_size, hidden_size, bias) for _ in range(num_hidden) ])
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.act(self.in_proj(x))

        for layer in self.hidden:
            x = self.act(layer(x))
        
        return self.out_proj(x)