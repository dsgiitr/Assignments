import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        #self.linear2 = nn.Linear(3, output_dim)

    def forward(self, x):
        out = self.linear1(x)
        #out = self.linear2(out)
        return out
