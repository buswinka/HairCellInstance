import torch
import torch.nn as nn
from src.modules import RDCBlock
from warnings import filterwarnings

# try:
#     from hcat.utils import pad_image_with_reflections
# except ModuleNotFoundError:
#     from HcUnet.utils import pad_image_with_reflections


filterwarnings("ignore", category=UserWarning)

class RDCNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RDCNet, self).__init__()

        complexity = 10

        self.strided_conv = nn.Conv3d(in_channels, complexity, kernel_size=3, stride=2, padding=1)
        self.RDCblock = RDCBlock(complexity)
        self.out_conv = nn.Conv3d(complexity, out_channels=complexity, kernel_size=3,padding=1)
        self.transposed_conv = nn.ConvTranspose3d(in_channels=complexity, out_channels=out_channels,
                                                  stride=(2,2,2), kernel_size=(4, 4, 4), padding=(1,1,1))

    def forward(self, x):
        x = self.strided_conv(x)
        for t in range(10):
            if t == 0:
                y = torch.zeros(x.shape).cuda()
            in_ = torch.cat((x, y), dim=1)
            y = self.RDCblock(in_) + y
        y = self.out_conv(y)
        return self.transposed_conv(y)
