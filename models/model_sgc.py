import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, num_features, nclass, device="cuda:0"):
        super(SGC, self).__init__()        
        self.device = device
        self.W = nn.Linear(num_features, nclass)
        
        self.to(device)

    def forward(self, x):
        return self.W(x)
