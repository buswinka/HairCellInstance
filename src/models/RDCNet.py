import torch
import torch.nn as nn
from src.models.modules.RDCblock import RDCBlock
from warnings import filterwarnings


filterwarnings("ignore", category=UserWarning)


class RDCNet(nn.Module):
    def __init__(self, in_channels, out_channels, complexity: int = 10):
        super(RDCNet, self).__init__()

        self.strided_conv = nn.Conv3d(in_channels, complexity, kernel_size=3, stride=2, padding=1)
        self.RDCblock = RDCBlock(in_channels=complexity)
        self.out_conv = nn.Conv3d(complexity, out_channels=out_channels, kernel_size=1, padding=0)
        self.transposed_conv = nn.ConvTranspose3d(in_channels=complexity, out_channels=complexity,
                                                  stride=(2, 2, 2), kernel_size=(4, 4, 4), padding=(1, 1, 1))
        self.batch_norm_out = nn.BatchNorm3d(out_channels)
        self.batch_norm_transpose = nn.BatchNorm3d(complexity)

        self.activation = nn.Tanh()

    def forward(self, x, i: int = 2):

        x = self.activation(self.strided_conv(x))

        for t in range(i):
            if t == 0:
                y = torch.zeros(x.shape).to(x.device)
            in_ = torch.cat((x, y), dim=1)
            y = self.RDCblock(in_) + y

        y = self.activation(self.batch_norm_transpose(self.transposed_conv(y)))
        return self.activation(self.batch_norm_out(self.out_conv(y)))

