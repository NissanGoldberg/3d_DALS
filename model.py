# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/model.py
# https://www.youtube.com/watch?v=IHq1t7NxS8k&t=1008s
from itertools import zip_longest

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from torchsummary import summary

GROWTH_K = 6
DROPOUT_RATE = 0.3


class DilationBlock(nn.Module):
    def __init__(self,  dilation_sz, in_channels_sz=294, filters=1024):
        super(DilationBlock, self).__init__()

        self.dilation_block = nn.Sequential(
            nn.Conv2d(in_channels_sz, filters, kernel_size=3, stride=1, padding=dilation_sz, dilation=dilation_sz, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters)
        )

    def forward(self, x):
        return self.dilation_block(x)


class Dilation(nn.Module):
    def __init__(self, in_channels_sz=294, dilation_sz=[6, 12, 18, 24]):
        super(Dilation, self).__init__()

        self.dilations = nn.ModuleList()

        for d in dilation_sz:
            self.dilations.append(DilationBlock(dilation_sz=d, in_channels_sz=in_channels_sz))

    def forward(self, x):
        # create 4 different dilation blocks and then concat
        y = []
        for dilation in self.dilations:
            y.append(dilation(x))

        x = torch.cat([*y], dim=1)
        return x # default: torch.Size([1, 4096, 8, 8])


class UpConv2d(nn.Module):
    def __init__(self, input_channels_sz, out_channels_sz ):
        super(UpConv2d, self).__init__()

        self.up_conv = nn.Sequential(
            nn.Conv2d(input_channels_sz, out_channels_sz, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels_sz),
            nn.Conv2d(out_channels_sz, out_channels_sz, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels_sz),
        )

    def forward(self, x):
        return self.up_conv(x)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=DROPOUT_RATE):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=DROPOUT_RATE):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)


class DenseBlock(nn.Module):
    def __init__(self,  nb_layers, in_planes, block=BottleneckBlock, growth_rate=GROWTH_K, dropRate=DROPOUT_RATE):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

# ========= START FROM HERE =======

class UNET(nn.Module):
    def __init__(
            self, dense_down_in_channels=[64, 6, 6, 6],
            trans_down_in_channels=[136, 150, 438],
            nb_layers=[3, 6, 18, 12],
            up_list=[4096, 512, 256, 128, 64],
            up_conv_in=[950, 406, 264, 128],

    ):
        super(UNET, self).__init__()
        self.ups_trans = nn.ModuleList()
        self.ups_conv = nn.ModuleList()
        self.downs_dense = nn.ModuleList()
        self.downs_trans = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # fist part
        self.conv_start = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.ReLU(inplace=True),
        )

        # Down part of UNet
        for dense_chan, nb_layer in zip(dense_down_in_channels, nb_layers):
            self.downs_dense.append(DenseBlock(in_planes=dense_chan, nb_layers=nb_layer, growth_rate=24))

        for trans_chan in trans_down_in_channels:
            self.downs_trans.append(TransitionBlock(in_planes=trans_chan, out_planes=6))

        self.dilation = Dilation()

        # upwards
        self.convT1 = nn.ConvTranspose2d(4096, 512, kernel_size=2, stride=2)

        for in_chan, out_chan in zip(up_list, up_list[1:]):
            self.ups_trans.append(nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2))


        up_conv_out = 512
        for up_in in up_conv_in:
            self.ups_conv.append(UpConv2d(input_channels_sz=up_in, out_channels_sz=up_conv_out))
            up_conv_out //= 2

        # downwards (end of UNet)
        self.end_of_UNet = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        skip_connections = []

        x = self.conv_start(x)
        skip_connections.append(x)

        x = self.pool(x)

        #  TODO fix, hack of None
        for down_dense, down_trans in zip_longest(self.downs_dense, self.downs_trans, fillvalue=None):
            x = down_dense(x)

            if down_trans!=None:
                skip_connections.append(x)
                x = down_trans(x)

        x = self.dilation(x)
        print("DDUnet shape:", x.shape)

        skip_connections = skip_connections[::-1]  # reverse list

        # upwards
        out_conv = 512
        for skip, up_trans, up_conv in zip(skip_connections, self.ups_trans, self.ups_conv):
            x = up_trans(x)
            x = torch.cat([x, skip], dim=1)
            x = up_conv(x)
            print(x.shape)
            out_conv /= 2

        # down again (end)
        x = self.end_of_UNet(x)
        print(x.shape)

        # TODO return lambdas
        map_lambda1 = torch.exp((2.0 - x)/(1.0 + x))
        map_lambda2 = torch.exp((1.0 + x)/(2.0 - x))
        return x


def test():
    x = torch.randn((1, 1, 256, 256))
    model = UNET()
    out_seg = model(x)
    # print(out_seg.shape)
    # print(x.shape)
    # print(model)
    # image = torch.rand((1, 572, 572))
    # summary(model, (1, 256, 256))
    # assert out_seg.shape == x.shape

    map_lambda1 = torch.exp((2.0 - out_seg) / (1.0 + out_seg))
    map_lambda2 = torch.exp((1.0 + out_seg) / (2.0 - out_seg))



if __name__ == "__main__":
    test()