import torch
import torch.nn as nn
from torch.nn.functional import max_pool2d
from models.layers import ConvX2
import unittest
import numpy as np


class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = ConvX2([3, 32, 32])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvX2([32, 64, 64])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvX2([64, 128, 128])
        self.pool3 = nn.MaxPool2d(2)
        return

    def forward(self, x):
        x = self.conv1(x)       # 3x64x64 -> 32x60x60
        x = self.pool1(x)       # 32x60x60 -> 32x30x30
        x = self.conv2(x)       # 32x30x30 -> 64x26x26
        x = self.pool2(x)       # 64x26x26 -> 64x13x13
        x = self.conv3(x)       # 64x13x13 -> 128x9x9
        x = self.pool3(x)       # 128x9x9 -> 128x4x4
        return x


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.do1 = torch.nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128*4*4, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.do2 = torch.nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 2)
        return

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.do1(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.do2(x)
        x = self.fc2(x)
        return x


class FaceDiscriminator(nn.Module):

    def __init__(self):
        super(FaceDiscriminator, self).__init__()
        self.fext = FeatureExtractor()
        self.clf = Classifier()
        self.act = nn.Sigmoid()
        return

    def forward(self, x):
        x = self.fext(x)
        logits = self.clf(x)
        probs = self.act(logits)
        return probs

    def __str__(self):
        return "FaceDiscriminator"


class TestDescriminator(unittest.TestCase):

    def test_forward_pass(self):
        batch_sz = 2
        model = FaceDiscriminator()
        test_in = torch.from_numpy(np.arange(64*64*3*batch_sz).reshape(batch_sz, 3, 64, 64).astype(np.float32))
        net_out = model.forward(test_in)
        self.assertEqual(net_out.shape, (batch_sz, 2))
        return

    def test_back_prop(self):
        batch_sz = 2
        model = FaceDiscriminator()

        input = torch.from_numpy(np.arange(64*64*3*batch_sz).reshape(batch_sz, 3, 64, 64).astype(np.float32))
        out = model.forward(input)
        target = torch.rand(batch_sz, 2)

        loss_func = nn.BCELoss()
        loss = loss_func(out, target)
        loss.backward()
        return


if __name__ == "__main__":
    unittest.main()
