import json
import models
import os
import torch

class ClassNames:
    dtype_double = "java.lang.Double"
    dtype_float = "java.lang.Float"
    dtype_bool = "java.lang.Boolean"
    dtype_int = "java.lang.Integer"
    dtype_long = "java.lang.Long"

class Utils:
    @staticmethod
    def is_linear_weight(model, state_dict, key):
        module_path = key.rsplit(".weight", 1)[0]
        module_path = module_path.rsplit(".bias", 1)[0]

        module = model
        for attr in module_path.split("."):
            if hasattr(module, attr):
                module = getattr(module, attr)
            else:
                return False

        return isinstance(module, torch.nn.Linear)

dtype_overwrite = {  }

def named(obj: any, name: str):
    return { name: obj }

def dtype(dtype: torch.dtype) -> str:
    asStr = str(dtype).removeprefix("torch.")
    
    if asStr in dtype_overwrite:
        return dtype_overwrite[asStr]

    if asStr.__contains__("float"):
        return "java.lang.Float"

    if asStr.__contains__("bool"):
        return "java.lang.Boolean"
    
    if dtype is torch.long:
        return "java.lang.Long"
    
    if dtype is torch.double:
        return "java.lang.Double"
    
    if asStr.__contains__("int"):
        return "java.lang.Integer"
    
    return "null"

def tensor(obj: torch.Tensor):
    return { 
        "dtype": dtype(obj.dtype),
        "shape": list(obj.shape),
        "data": obj.detach().flatten().tolist()
    }

def linear(obj: torch.nn.Linear):
    out = { "weight": tensor(obj.weight.T) }
    
    if obj.bias is not None:
        out["bias"] = tensor(obj.bias)
    
    return out

def embedding(obj: torch.nn.Embedding):
    return { "weight": tensor(obj.weight) }

def layer_norm(obj: torch.nn.LayerNorm):
    return { "epsilon": obj.eps, "weight": tensor(obj.weight), "bias": tensor(obj.bias) }

def parameter(param: torch.nn.Parameter):
    return tensor(param.data)

def from_state(state):
    return { k: parameter(v) for k, v in zip(state.keys(), state.values()) }

def from_module(module: torch.nn.Module):
    state = module.state_dict()
    out = {}

    for k, v in zip(state.keys(), state.values()):
        if Utils.is_linear_weight(module, state, k):
            print("transposing {}".format(k))
            v.data = v.data.T

        out[k] = parameter(v)

    return out

def export(data, file=None, sort=True, pretty=False, indent=4):
    args = { "obj": data, "sort_keys": sort }

    if pretty:
        args["indent"] = indent

    if file is not None:
        with open(file, "w+") as f:
            args["fp"] = f
            return json.dump(**args)

    else:
        return json.dumps(**args)

def append(root: dict, new: dict, key: str = ""):
    if key:
        key = key + "."

    for k, v in new.items():
        root[key + k] = v


def transformer(model: models.Transformer):
    out = {}

    # append(out, named(model.block_size, "block_size"))
    append(out, linear(model.lm_head),  "lm_head")
    append(out, layer_norm(model.ln_f), "ln_f")
    append(out, embedding(model.wte),   "wte")
    append(out, embedding(model.wpe),   "wpe")

    for i, head in enumerate(model.heads):        
        append(out, linear(head.attn.c_attn), f"heads.{i}.attn.c_attn")
        append(out, linear(head.attn.c_proj), f"heads.{i}.attn.c_proj")
        # append(out, named(head.attn.n_embd, "n_embd"), f"heads.{i}.attn")
        # append(out, named(head.attn.n_head, "n_head"), f"heads.{i}.attn")
        append(out, layer_norm(head.ln_1),    f"heads.{i}.ln_1")
        append(out, layer_norm(head.ln_2),    f"heads.{i}.ln_2")
        append(out, linear(head.mlp.c_fc),    f"heads.{i}.mlp.modules.0")
        append(out, linear(head.mlp.c_proj),    f"heads.{i}.mlp.modules.2")
    
    return out

def lstm(model: torch.nn.LSTM):
    class fL():
        def __init__(self, weight, bias):
            self.weight = weight
            self.bias = bias

    out = {}

    sd = model.state_dict()

    for i in range(model.num_layers):
        append(out, linear(fL(sd[f"weight_ih_l{i}"], sd[f"bias_ih_l{i}"])), f"cells.{i}.ih")
        append(out, linear(fL(sd[f"weight_hh_l{i}"], sd[f"bias_hh_l{i}"])), f"cells.{i}.hh")
    
    return out

def lstm_for_nlp(model: models.LSTMForNLP):
    out = {}

    append(out, embedding(model.embedding), "embedding")
    append(out, lstm(model.lstm),           "lstm")
    append(out, linear(model.out_proj),     "out_proj")

    return out