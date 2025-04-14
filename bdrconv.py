import torch
import torch.nn as nn
import math

class BDRConv(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, dw_size=3, stride=1, in_ratio=2, out_ratio=2, exp=2, relu=True,
                 split=True, reduction=16):
        super(BDRConv, self).__init__()
        self.split = split
        self.oup = oup
        self.stride = stride
        self.in_ratio = in_ratio

        init_channels = int(math.ceil(oup / out_ratio))
        new_channels = init_channels * (out_ratio - 1)

        if split:
            exp_in = inp // in_ratio
            self.main_in = inp - exp_in
        else:
            self.main_in = inp
            exp_in = inp

        # Main branch
        self.main_part = nn.Conv2d(
            in_channels=self.main_in,
            out_channels=init_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False
        )

        # Expansion branch
        self.expand_operation = nn.Conv2d(
            in_channels=exp_in,
            out_channels=init_channels * exp,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.Relu = nn.ReLU(inplace=True)

        detail_input_channels = self.main_in + init_channels * exp

        self.detail_part = nn.Conv2d(
            detail_input_channels,
            new_channels,
            dw_size,
            stride=1,
            padding=dw_size // 2,
            groups=self.main_in,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(detail_input_channels)
        self.bn2 = nn.BatchNorm2d(self.oup)

        self.avgpool_s2_detail = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgpool_s2_expand = nn.AvgPool2d(kernel_size=2, stride=2)

        # Squeeze-and-Excitation block
        self.seB = 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        oup = oup // self.seB
        self.fc = nn.Sequential(
            nn.Linear(oup, max(2, oup // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(2, oup // reduction), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_main = x[:, :self.main_in, :, :]
        x_expand = x[:, self.main_in:, :, :]

        x1 = self.main_part(x_main)

        if self.stride == 2:
            x_main = self.avgpool_s2_detail(x_main)
            x_expand = self.avgpool_s2_expand(x_expand)

        exp = self.expand_operation(x_expand)

        detail_in = torch.cat([x_main, exp], dim=1)
        detail_in = self.bn1(detail_in)
        x2 = self.detail_part(detail_in)

        out = torch.cat([x1, x2], dim=1)
        out = self.bn2(out)

        # SE block
        b, c, _, _ = out.size()
        y = self.avg_pool(out).view(b * self.seB, c // self.seB)
        y = self.fc(y).view(b, c, 1, 1)
        out = out * y

        # Residual connection
        out += exp

        return out[:, :self.oup, :, :]
