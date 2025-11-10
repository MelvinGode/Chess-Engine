import torch
import torch.nn.functional as F
import numpy as np


legal_moves = np.array([[15,30]])
print(legal_moves)
selected_start=15
print(legal_moves[np.isclose(legal_moves[:,0], selected_start), 1])
