import torch.nn as nn
from torch.nn.init import kaiming_normal_ as he_normal


class ConvX1(nn.Module):

    def __init__(self, channels):
        super(ConvX1, self).__init__()
        self.conv = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels[1])
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class UpSample(nn.Module):

    def __init__(self, channels, stride=2, kernel=5):
        super(UpSample, self).__init__()
        assert len(channels) == 2
        in_channel, out_channel = channels
        self.upsample = nn.ConvTranspose2d(in_channels=in_channel,
                                           out_channels=out_channel,
                                           kernel_size=kernel,
                                           stride=stride,
                                           padding=0,
                                           bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        in_shape = x.shape
        target_ysz, target_xsz = 2 * in_shape[2], 2 * in_shape[3]
        x = self.act(self.bn(self.upsample(x)))
        offset_x, offset_y = (x.shape[3] - target_xsz) // 2, (x.shape[2] - target_ysz) // 2
        return x[:, :, offset_y:offset_y+target_ysz, offset_x:offset_x+target_xsz]
