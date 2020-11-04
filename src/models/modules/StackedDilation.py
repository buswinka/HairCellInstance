import torch
import torch.nn as nn
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning)


class StackedDilation(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: tuple,
                 dilation: tuple = None,
                 padding: tuple = None
                 ):

        super(StackedDilation, self).__init__()

        self.dilation = dilation if dilation is not None else (1, 2, 3, 4, 5)
        self.padding = padding if padding is not None else (2, 4, 6, 8, 10)

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel, dilation=1, padding=2)
        self.out_conv = nn.Conv3d(out_channels*len(self.dilation), out_channels, kernel_size=1, padding=0)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        for i, (dilation, padding) in enumerate(zip(self.dilation, self.padding)):
            self.conv.dilation = dilation
            self.conv.padding = padding
            x = self.relu(self.conv(x))
            if i == 0:
                out = x
            else:
                out = torch.cat((out, x), dim=1)
        out = self.relu(self.out_conv(out))
        return out
