import torch
import torch.nn as nn
from src.models.modules.RDCblock import RDCBlock
import src.functional
from warnings import filterwarnings


filterwarnings("ignore", category=UserWarning)


class RDCNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, complexity: int = 20):
        super(RDCNet, self).__init__()

        self.conv3x3_1 = nn.Conv3d(in_channels=in_channels, out_channels=5, padding=1, kernel_size=3)

        self.strided_conv = nn.Conv3d(5, complexity, kernel_size=3, stride=2, padding=1)
        self.RDCblock = RDCBlock(in_channels=complexity)
        self.transposed_conv = nn.ConvTranspose3d(in_channels=complexity, out_channels=complexity,
                                                  stride=(2, 2, 2), kernel_size=(4, 4, 4), padding=(1, 1, 1))
        self.out_conv = nn.Conv3d(complexity, out_channels=out_channels, kernel_size=1, padding=0)


        self.batch_norm_out = nn.BatchNorm3d(out_channels)
        self.batch_norm_transpose = nn.BatchNorm3d(complexity)

        self.bn_1 = nn.BatchNorm3d(5)

        self.activation = nn.LeakyReLU()

    def forward(self, x, i: int = 7):
        x = self.activation(self.bn_1(self.conv3x3_1(x)))
        x = self.activation(self.strided_conv(x))
        y = torch.zeros(x.shape).to(x.device)

        for t in range(i):
            in_ = torch.cat((x, y), dim=1)
            y = self.RDCblock(in_) + y

        y = self.activation(self.batch_norm_transpose(self.transposed_conv(y)))
        y = self.out_conv(y)

        if self.training:
            return y
        else:
            y = src.functional.vector_to_embedding(y)
            centroids = src.functional.estimate_centroids(y).unsqueeze(0)
            return src.functional.embedding_to_probability(y, centroids, torch.tensor([0.01])), y, centroids