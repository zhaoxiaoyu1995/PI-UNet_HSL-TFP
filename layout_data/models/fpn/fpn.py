# -*- coding: utf-8 -*-
# @Time    : 2021/8/31 9:24
# @Author  : zhaoxiaoyu
# @File    : fpn.py
from layout_data.models.fpn.resnet import resnet18
from layout_data.models.fpn.fpn_head import FPNDecoder
import torch.nn as nn
import torch


class fpn(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18()
        self.head = FPNDecoder(encoder_channels=[512, 256, 128, 64])

    def forward(self, input):
        x = self.backbone(input)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = fpn().cuda()
    print(model)
    x = torch.zeros(1, 1, 200, 200).cuda()
    with torch.no_grad():
        y = model(x)
    print(y.size())