import torch.nn as nn

class LSTMForNLP(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, hidden_size: int, num_layers: int, bias: bool):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.lstm = nn.LSTM(n_embd, hidden_size, num_layers, bias)
        self.out_proj = nn.Linear(hidden_size, vocab_size)
        self.last_hx = None
    
    def forward(self, idx, hx = None):
        hx = hx if hx != None else self.last_hx

        e = self.embedding(idx)
        idx, hx = self.lstm(e, hx)
        self.last_hx = hx
        return self.out_proj(idx)