import torch
import torch.nn as nn
from torch.nn import functional as F


class StructureUpsampling(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(2, 32, kernel_size=7, stride=1, padding=0),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1))

    def forward(self, edge, line):
        x = torch.cat([edge, line], dim=1)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.convs(x)
        edge_out = x[:, 0:1, ...]
        line_out = x[:, 1:2, ...]

        return edge_out, line_out


class StructureUpsampling2(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(2, 64, kernel_size=7, stride=1, padding=0),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1))

    def forward(self, edge, line):
        x = torch.cat([edge, line], dim=1)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.convs(x)
        edge_out = x[:, 0:1, ...]
        line_out = x[:, 1:2, ...]

        return edge_out, line_out


class StructureUpsampling3(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=0),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, line):
        x = line
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.convs(x)

        return x


class StructureUpsampling4(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=0),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))
        self.out = nn.Sequential(nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, line):
        x = line
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.convs(x)
        x2 = self.out(x)

        return x, x2
