import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, bias: bool):
        super().__init__()

        self.n_head = n_head
        self.n_embd = n_embd

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                            .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # batch, seq_len, n_embd

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        qk = (q @ k.transpose(-2, -1)) * (1.0 / torch.tensor(k.size(-1)).sqrt())
        print(x.flatten())
        qk = qk.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # apply mask
        qkv = F.softmax(qk, dim=-1) @ v

        y = qkv.transpose(1, 2).contiguous().view(B, T, C) # reassemble heads

        return y

class MLP(nn.Module):
    def __init__(self, n_embd: int, bias: bool):
        super().__init__()

        self.c_fc = nn.Linear(n_embd, n_embd * 4, bias)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(n_embd * 4, n_embd, bias)
    
    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, bias: bool):
        super().__init__()

        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, bias)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_layer: int, block_size: int, vocab_size: int, bias: bool):
        super().__init__()

        self.block_size = block_size

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.heads = nn.ModuleList([ Block(n_embd, n_head, block_size, bias) for _ in range(n_layer) ])
        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size, False)
    
    def forward(self, idx):
        B, T = idx.size() # batch size, seq length

        assert T <= self.block_size, "Sequence too long!"

        x = self.wte(idx) + self.wpe(torch.arange(0, T, device = idx.device))

        for block in self.heads:
            x = block(x)
        
        x = self.ln_f(x)

        logits = self.lm_head(x) # b, t, voc_size

        return logits

    @staticmethod
    def from_pretrained(model_type = "distilgpt2"):
        from transformers import GPT2LMHeadModel

        # get model args based on model type
        args = {
            'distilgpt2':   dict(n_layer=6, n_head=12, n_embd=768),   # 82M params
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        args['vocab_size'] = 50257
        args['block_size'] = 1024
        args['bias'] = True

        print("Using args: " + str(args))

        # init random model
        model = Transformer(**args)

        # init a huggingface/transformers model
        gpt = GPT2LMHeadModel.from_pretrained(model_type)
    
        transposed = [ "attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight" ]
        ignored = [ ".attn.bias", ".attn.masked_bias" ]

        # get model state
        model_sd = model.state_dict()
        model_sd_keys = [ k for k in model_sd.keys() if not any([k.endswith(x) for x in ignored]) ]

        # get gpt model state
        gpt_sd = gpt.state_dict()
        gpt_sd_keys = [ k for k in gpt_sd.keys() if not any([k.endswith(x) for x in ignored]) ]

        # transfer weights
        for x, y in zip(gpt_sd_keys, model_sd_keys):
            if any([x.endswith(p) for p in transposed]):
                model_sd[y] = gpt_sd[x].transpose(-2, -1)
            else:
                model_sd[y] = gpt_sd[x]
        
        model.load_state_dict(model_sd)

        return model

    @torch.no_grad
    def generate(self, idx, max_new_tokens = 100, temperature = 1.0, top_k = None, eos_token_id = None):
        idx = idx.to(next(self.parameters()).device)

        if len(idx[-1, :]) == 0:
            return idx
        
        for _ in range(max_new_tokens):
            _idx = idx.clone()
            logits = self(_idx)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)

            if eos_token_id is not None and eos_token_id == next_token:
                break
        
        return idx
    
    def prompt(self, prompt, tokenizer, max_new_tokens = 100):
        x = tokenizer.encode(prompt, return_tensors="pt")
        return tokenizer.decode(self.generate(x, max_new_tokens, 1.0, None, tokenizer.eos_token_id).squeeze())