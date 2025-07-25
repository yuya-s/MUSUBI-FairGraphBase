import torch
import numpy as np

def run_fairdrop(data, args, trial=1):
    edge_index = data.edge_index
    sens = data.sens
    device = sens.device
    delta = getattr(args, 'fairdrop_delta', 0.1) # Default value for delta is 0.1
    src, dst = edge_index
    y_aux = (sens[src] != sens[dst])
    E = y_aux.size(0)
    randomizer = torch.rand(1, E, device=device) < (0.5 + delta)
    keep = torch.where(randomizer[0], y_aux, ~y_aux)
    kept_edges = keep.sum().item()
    total_edges = E
    data.edge_index = edge_index[:, keep]