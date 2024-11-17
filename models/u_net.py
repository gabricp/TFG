import torch.nn as nn
import numpy as np
import torch


class UNET(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

    def __len__(self):
        return len(self.X)

    def forward(self,x):
        return x 

if "__name__" == "__main__":
    model = UNET(input_shape=(1, 256, 256))
    