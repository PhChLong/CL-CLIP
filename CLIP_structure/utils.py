import torch
import torch.nn.functional as F
def _get_vector_norm(x: torch.Tensor):
    return x.norm(p=2, dim = -1, keepdim=True).clamp(1e-8)
def create_causal_mask():
    pass
def clip_loss():
    pass