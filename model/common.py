from os import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=1, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    '''

    args:

    '''
    def __init__(self,
                 inc: int,
                 n: int=1,
                 g=1,
                 e=0.5,
                 p=1,
                 shortcut=True):
        super().__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_channels=inc,
                      out_channels=inc // 2,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      dilation=1,
                      groups=1,
                      bias=False),
            nn.BatchNorm2d(inc // 2),
            nn.SiLU(inplace=True),
        )

        # self.cv2 = nn.Sequential(
        #
        # )
        #
        # self.cv3 = nn.Sequential(
        #
        # )  # optional act=FReLU(c2)
        #
        # self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        #
        # for _ in range(n):
        #     pass

    def forward(self, x):
        # return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
        return self.cv1(x)

