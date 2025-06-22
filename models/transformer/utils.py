import copy
from torch import nn

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])