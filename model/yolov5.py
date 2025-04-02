import torch
import torch.nn as nn
import torch.nn.functional as F

from common import *


class ResidualBlock(nn.Module):
    def __init__(self, outc: int, k: list, s: list):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(outc, outc, kernel_size=k[0], stride=s[0], bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc, outc, kernel_size=k[1], stride=s[0], bias=False),
            nn.BatchNorm2d(outc)
        )

    def forward(self, x):
        return self.conv_block(x)

class CSP_Bottleneck(nn.Module):
    def __init__(self, inc: int=1, outc: int=1, k: list=(1, 3), s: list=(1, 2), n: int=1):
        super().__init__()

        self.cv1 = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=k[0], stride=s[0], bias=False),
            nn.BatchNorm2d(outc, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.cv2 = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=k[0], stride=s[0], bias=False),
            nn.BatchNorm2d(outc, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.cv3 = nn.Sequential(
            nn.Conv2d(inc, inc, kernel_size=k[0], stride=s[0], bias=False),
            nn.BatchNorm2d(inc, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.ml = nn.Sequential(*[ResidualBlock(outc, k, s) for _ in range(n)])

    def forward(self, x):

        x = self.ml(self.cv1(x))
        return x
        # return self.cv3(torch.cat((self.ml(self.cv1(x), self.cv2(x))), 1))

class YOLOv5(nn.Module):
    def __init__(self):
        super(YOLOv5, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.layer3 = CSP_Bottleneck(64, 32, [1, 3], [1, 2])

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        return x



if __name__ == '__main__':
    data = torch.randn(1, 64, 160, 160)
    model = CSP_Bottleneck(64, 32, [1, 3], [1, 2])
    out = model(data)
    print(out.shape)
