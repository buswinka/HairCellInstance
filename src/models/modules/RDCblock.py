import torch
import torch.nn as nn
from src.models.modules.StackedDilation import StackedDilation


from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning)


class RDCBlock(nn.Module):
    def __init__(self, in_channels):

        super(RDCBlock, self).__init__()

        self.conv = nn.Conv3d(in_channels*2, in_channels, kernel_size=1)
        self.grouped_conv = StackedDilation(in_channels, in_channels, 5)

    def forward(self, x):
        x = self.conv(x)
        x = self.grouped_conv(x)
        return x
