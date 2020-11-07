import torch
import torch.nn as nn
from models.layers import UpSample
import unittest


class FaceGenerator(nn.Module):

    def __init__(self, feature_vector=100):
        super(FaceGenerator, self).__init__()
        self.fc = nn.Linear(feature_vector, 4 * 4 * 512, bias=False)
        self.bn_fc = nn.BatchNorm1d(4 * 4 * 512)
        self.act_lrelu = nn.LeakyReLU(negative_slope=0.2)

        self.up1 = UpSample(channels=[512, 512], stride=1)  # 4x4x512 -> 8x8x512
        self.up2 = UpSample(channels=[512, 256], stride=2)  # 8x8x512 -> 16x16x256
        self.up3 = UpSample(channels=[256, 128], stride=2)  # 16x16x256 -> 32x32x128

        self.deconv = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=5, stride=2, padding=0,
                                         bias=False)
        self.bn_final = nn.BatchNorm2d(3)
        self.act_tanh = nn.Tanh()

    def forward(self, x):
        x = self.act_lrelu(self.bn_fc(self.fc(x)))  # 100 -> 4*4*512
        x = x.view(-1, 512, 4, 4)                   # 4*4*512 -> 512x4x4
        x = self.up1(x)                             # 512x4x4 -> 512x8x8
        x = self.up2(x)                             # 512x8x8 -> 256x16x16
        x = self.up3(x)                             # 256x16x16 -> 128x32x32
        x = self.act_tanh(self.bn_final(self.deconv(x)))    # 128x32x32 -> 3x64x64
        return x[:, :, 1:1+64, 1:1+64]

    def __str__(self):
        return "FaceGenerator"


class TestGenerator(unittest.TestCase):

    def test_forward(self):
        batch_sz = 2
        model = FaceGenerator()
        test_in = torch.rand(batch_sz, 100)
        net_out = model.forward(test_in)
        self.assertEqual(net_out.shape, (batch_sz, 3, 64, 64))

    def test_backward(self):
        batch_sz = 2
        model = FaceGenerator()
        test_in = torch.rand(batch_sz, 100)
        net_out = model.forward(test_in)
        test_out = torch.rand(batch_sz, 3, 64, 64)
        loss = nn.MSELoss()
        output = loss(net_out, test_out)
        output.backward()
        print(output.item())


if __name__ == "__main__":
    unittest.main()
