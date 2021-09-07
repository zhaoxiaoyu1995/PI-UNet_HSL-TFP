import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, size):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return x


class FCN8s(nn.Module):
    def __init__(self, num_classes, in_channels=3, bn=False):
        super(FCN8s, self).__init__()
        vgg = models.vgg16()
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        if in_channels != 3:
            features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        features_temp = []
        for i in range(len(features)):
            features_temp.append(features[i])
            if isinstance(features[i], nn.Conv2d):
                features_temp.append(
                    nn.BatchNorm2d(features[i].out_channels) if bn else
                    nn.GroupNorm(32, features[i].out_channels))

        features = features_temp
        self.features3 = nn.Sequential(*features[:24])
        self.features4 = nn.Sequential(*features[24: 34])
        self.features5 = nn.Sequential(*features[34:])

        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)

        fc6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        fc7 = nn.Conv2d(512, 512, kernel_size=1)
        score_fr = nn.Conv2d(512, num_classes, kernel_size=1)

        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True), score_fr
        )
        self.upscore2 = Conv3x3GNReLU(num_classes, num_classes, upsample=True)
        self.upscore_pool4 = Conv3x3GNReLU(num_classes, num_classes, upsample=True)
        self.final_conv = nn.Conv2d(num_classes, 1, kernel_size=1)

    def forward(self, x):
        pool3 = self.features3(x)
        pool4 = self.features4(pool3)
        pool5 = self.features5(pool4)

        score_fr = self.score_fr(pool5)
        upscore2 = self.upscore2(score_fr, pool4.size()[-2:])

        score_pool4 = self.score_pool4(pool4)
        upscore_pool4 = self.upscore_pool4(score_pool4 + upscore2, pool3.size()[-2:])

        score_pool3 = self.score_pool3(pool3)
        upscore8 = F.interpolate(self.final_conv(score_pool3 + upscore_pool4), x.size()[-2:], mode='bilinear', align_corners=True)
        return upscore8


class FCN16s(nn.Module):
    def __init__(self, num_classes):
        super(FCN16s, self).__init__()
        vgg = models.vgg16()
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features4 = nn.Sequential(*features[: 24])
        self.features5 = nn.Sequential(*features[24:])

        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()

        fc6 = nn.Conv2d(512, 4096, kernel_size=3, padding=1)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )
        self.upscore2 = Conv3x3GNReLU(num_classes, num_classes, upsample=True)
        self.upscore16 = Conv3x3GNReLU(1, 1, upsample=True)
        self.final_conv = nn.Conv2d(num_classes, 1, kernel_size=1)

    def forward(self, x):
        pool4 = self.features4(x)
        pool5 = self.features5(pool4)

        score_fr = self.score_fr(pool5)
        upscore2 = self.upscore2(score_fr, pool4.size()[-2:])

        score_pool4 = self.score_pool4(0.1 * pool4)
        upscore16 = self.upscore16(self.final_conv(score_pool4 + upscore2), x.size()[-2:])
        return upscore16


class FCN32s(nn.Module):
    def __init__(self, num_classes):
        super(FCN32s, self).__init__()
        vgg = models.vgg16()
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features5 = nn.Sequential(*features)

        fc6 = nn.Conv2d(512, 4096, kernel_size=3, padding=1)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )
        self.upscore = Conv3x3GNReLU(1, 1, upsample=True)
        self.final_conv = nn.Conv2d(num_classes, 1, kernel_size=1)

    def forward(self, x):
        pool5 = self.features5(x)

        score_fr = self.score_fr(pool5)
        upscore = self.upscore(self.final_conv(score_fr), x.size()[-2:])
        return upscore


class FCNAlex8s(nn.Module):
    def __init__(self, num_classes, in_channels=3, bn=False):
        super(FCNAlex8s, self).__init__()

        self.features3 = nn.Sequential(
            # kernel(11, 11) -> kernel(7, 7)
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            # padding=0 -> padding=1
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.features5 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )

        self.score_pool3 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(192, num_classes, kernel_size=1)

        fc6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        fc7 = nn.Conv2d(512, 512, kernel_size=1)
        score_fr = nn.Conv2d(512, num_classes, kernel_size=1)

        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True), score_fr
        )
        self.upscore2 = Conv3x3GNReLU(num_classes, num_classes, upsample=True)
        self.upscore_pool4 = Conv3x3GNReLU(num_classes, num_classes, upsample=True)
        self.final_conv = nn.Conv2d(num_classes, 1, kernel_size=1)

    def forward(self, x):
        pool3 = self.features3(x)
        pool4 = self.features4(pool3)
        pool5 = self.features5(pool4)

        score_fr = self.score_fr(pool5)
        upscore2 = self.upscore2(score_fr, pool4.size()[-2:])

        score_pool4 = self.score_pool4(pool4)
        upscore_pool4 = self.upscore_pool4(score_pool4 + upscore2, pool3.size()[-2:])

        score_pool3 = self.score_pool3(pool3)
        upscore8 = F.interpolate(self.final_conv(score_pool3 + upscore_pool4), x.size()[-2:], mode='bilinear',
                                 align_corners=True)
        return upscore8


if __name__ == '__main__':
    model = FCN8s(in_channels=1, num_classes=128)
    print(model)
    x = torch.randn(1, 1, 200, 200)
    with torch.no_grad():
        y = model(x)
        print(y.shape)