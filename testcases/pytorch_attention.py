import torch
import torch.nn as nn
import torch.nn.functional as F
class CustomModel(nn.Module):
    def __init__(self, kwargs):
        super(CustomModel, self).__init__()
        self.kwargs = kwargs
        self.attn = nn.MultiheadAttention(embed_dim=kwargs['embedding_dim'], num_heads=kwargs['num_heads'], dropout=kwargs['dropout'], add_bias_kv=kwargs['add_bias_kv'], add_zero_attn=kwargs['add_zero_attn'], kdim=kwargs['kdim'], vdim=kwargs['vdim'], batch_first=kwargs['batch_first'])
    def forward(self, *args):
        query, key, value, attn_mask, kp_mask = args[0], args[1], args[2], args[3], args[4]
        return self.attn(query, key, value, attn_mask=attn_mask, key_padding_mask=kp_mask, need_weights=self.kwargs['need_weights'], average_attn_weights=self.kwargs['average_attn_weights'], is_causal=self.kwargs['is_causal'])

# Create model instance
model = CustomModel(kwargs = {
    'embedding_dim': 64,
    'num_heads': 1,
	'dropout': 0.1,
	'add_bias_kv': True,
    'add_zero_attn': False,
    'kdim': 16,
    'vdim': None, #used None inseatd of string(missing)
    'batch_first': True,
    'need_weights': False,
    'average_attn_weights': True,
    'is_causal': False
	})
	

# Dummy input tensors
query = torch.rand(1, 50, 64)         # (batch, seq_len, embedding_dim)
key = torch.rand(1, 10, 16)
value = torch.rand(1, 10, 64)
attn_mask = torch.zeros(50, 10)       # (seq_len, seq_len)
key_padding_mask = torch.zeros(1, 10)  # (batch, seq_len)


# Export the model
exported_model = torch.export.export(
    model,
    args=(query, key, value, attn_mask, key_padding_mask)
)

# use exported_model.graph to inspect the TorchScript graph
print(exported_model)

# after building torch_mlir
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType

mlir_model = fx.export_and_import(exported_model, output_type=OutputType.TORCH, experimental_support_mutation=True)

print("after fx.export_and_import")
print(mlir_model)