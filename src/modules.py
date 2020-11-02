import torch
import torch.nn as nn
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning)

@torch.jit.script
def crop(x, y):
    """
    Cropping Function to crop tensors to each other. By default only crops last 2 (in 2d) or 3 (in 3d) dimensions of
    a tensor.
    :param x: Tensor to be cropped
    :param y: Tensor by who's dimmension will crop x
    :return:
    """
    shape_x = x.shape
    shape_y = y.shape
    cropped_tensor = torch.empty(0)

    assert shape_x[1] == shape_y[1],\
        f'Inputs do not have same number of feature dimmensions: {shape_x} | {shape_y}'

    if len(shape_x) == 4:
        cropped_tensor = x[:, :, 0:shape_y[2]:1, 0:shape_y[3]:1]
    if len(shape_x) == 5:
        cropped_tensor = x[:, :, 0:shape_y[2]:1, 0:shape_y[3]:1, 0:shape_y[4]:1]

    return cropped_tensor


class Down(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: dict,
                 dilation: dict,
                 groups: dict,
                 padding=None
                 ):
        super(Down, self).__init__()
        if padding is None:
            padding = 0

        self.conv1 = nn.Conv3d(in_channels,
                                       out_channels,
                                       kernel['conv1'],
                                       dilation=dilation['conv1'],
                                       groups=groups['conv1'],
                                       padding=padding)

        self.conv2 = nn.Conv3d(out_channels,
                                       out_channels,
                                       kernel['conv2'],
                                       dilation=dilation['conv2'],
                                       groups=groups['conv2'],
                                       padding=1)

        self.batch1 = nn.BatchNorm3d(out_channels)
        self.batch2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.batch1(self.conv1(x)))
        x = self.relu(self.batch2(self.conv2(x)))
        return x


class Up(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: tuple,
                 upsample_kernel: tuple,
                 upsample_stride: int,
                 dilation: dict,
                 groups: dict,
                 padding_down=None,
                 padding_up=None
                 ):

        super(Up, self).__init__()

        if padding_down is None:
            padding_down=0
        if padding_up is None:
            padding_up=0

        self.conv1 = nn.Conv3d(in_channels,
                                   out_channels,
                                   kernel['conv1'],
                                   dilation=dilation['conv1'],
                                   groups=groups['conv1'],
                                   padding=padding_down)
        self.conv2 = nn.Conv3d(out_channels,
                                   out_channels,
                                   kernel['conv2'],
                                   dilation=dilation['conv2'],
                                   groups=groups['conv2'],
                                   padding=padding_down)

        self.up_conv = nn.ConvTranspose3d(in_channels,
                                         out_channels,
                                         upsample_kernel,
                                         stride=upsample_stride,
                                         padding=padding_up)
        self.lin_up = False

        self.batch1 = nn.BatchNorm3d(out_channels)
        self.batch2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        x = self.up_conv(x)
        y = crop(x, y)
        x = torch.cat((x, y), dim=1)
        x = self.relu(self.batch1(self.conv1(x)))
        x = self.relu(self.batch2(self.conv2(x)))
        return x


class StackedDilation(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: tuple,
                 ):

        super(StackedDilation, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel, dilation=1, padding=2)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel, dilation=2, padding=4)
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel, dilation=3, padding=6)
        self.conv4 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel, dilation=4, padding=8)
        self.conv5 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel, dilation=5, padding=10)
        self.out_conv = nn.Conv3d(out_channels*5, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)

        out = torch.cat((x1, x2, x3, x4, x5),dim=1)
        out = self.out_conv(out)
        return out


class RDCBlock(nn.Module):
    def __init__(self, in_channels):

        super(RDCBlock, self).__init__()

        self.conv = nn.Conv3d(in_channels*2, in_channels, kernel_size=1)
        self.grouped_conv = StackedDilation(in_channels, in_channels, 5)

    def forward(self, x):
        x = self.conv(x)
        x = self.grouped_conv(x)
        return x

