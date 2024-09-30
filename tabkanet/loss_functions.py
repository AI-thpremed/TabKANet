import torch
import torch.nn as nn

class RootMeanSquaredLogarithmicError(nn.Module):
    def __init__(self):
        super(RootMeanSquaredLogarithmicError, self).__init__()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean((torch.log1p(y_true) - torch.log1p(y_pred))**2))