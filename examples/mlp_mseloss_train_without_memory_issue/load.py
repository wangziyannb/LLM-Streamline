import torch
import copy

# model = torch.load('model.bin')
# torch.save({'config': copy.deepcopy(model.replace_layer.config), 'u_pruned': copy.deepcopy(model.replace_layer.up_proj),
#             'g_pruned': copy.deepcopy(model.replace_layer.gate_proj),
#             'd_pruned': copy.deepcopy(model.replace_layer.down_proj)}, 'outlier_mlp.pth')

model = torch.load('outlier_mlp.pth')
print()
