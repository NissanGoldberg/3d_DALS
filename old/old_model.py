# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/model.py
# https://www.youtube.com/watch?v=IHq1t7NxS8k&t=1008s

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

# class Bottleneck_Layer(nn.Module):
#     def __init__(self, in_channels):
#         super(Bottleneck_Layer, self).__init__()
#         self.relu = nn.ReLU(inplace=True)
#
#         self.bottleneck_layer = nn.Sequential(
#             nn.BatchNorm2d(in_channels),
#             nn.Conv2d(in_channels, 4*GROWTH_K, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=DROPOUT_RATE),
#             nn.BatchNorm2d(4*GROWTH_K),
#             nn.Conv2d(4*GROWTH_K, 4 * GROWTH_K, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=DROPOUT_RATE),
#         )
#
#     def forward(self, x):
#         return x
#
# class Dense_Block(nn.Module):
#     def __init__(self, in_channels, nb_layers):
#         super(Dense_Block, self).__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.nb_layers = nb_layers
#
#         # self.layers_concat = nn.ModuleList()
#
#         # self.bottleneck_layer = nn.Sequential(
#         #     nn.BatchNorm2d(in_channels),
#         #     nn.Conv2d(in_channels, 4*GROWTH_K, kernel_size=3, stride=1, padding=1, bias=False),
#         #     nn.ReLU(inplace=True),
#         #     nn.Dropout(p=DROPOUT_RATE),
#         #     nn.BatchNorm2d(4*GROWTH_K),
#         #     nn.Conv2d(4*GROWTH_K, 4 * GROWTH_K, kernel_size=3, stride=1, padding=1, bias=False),
#         #     nn.ReLU(inplace=True),
#         #     nn.Dropout(p=DROPOUT_RATE),
#         # )
#         self.bottleneck_layer = Bottleneck_Layer
#
#     def forward(self, input_x):
#         layers_concat = [input_x]
#         # print(input_x)
#         #
#         # for down in self.downs:
#         #     x = down(x)
#         #     skip_connections.append(x)
#         #     x = self.pool(x)
#         x = self.bottleneck_layer(input_x)
#         layers_concat.append(x)
#
#         # print(x.shape)
#         # print(input_x.shape)
#
#         for i in range(self.nb_layers-1):
#             x = torch.cat(layers_concat, 1) # 1 - for concat on dim 1
#             x = self.bottleneck_layer(x)
#             layers_concat.append(x)
#
#         x = torch.cat(layers_concat)
#         return x





class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], # each step
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dense1 = DenseBlock(in_planes=64, nb_layers=3, growth_rate=24)
        self.trans1 = TransitionBlock(in_planes=136, out_planes=6)
        self.dense2 = DenseBlock(in_planes=6, nb_layers=6, growth_rate=24)
        self.trans2 = TransitionBlock(in_planes=150, out_planes=6)
        self.dense3 = DenseBlock(in_planes=6, nb_layers=18, growth_rate=24)
        self.trans3 = TransitionBlock(in_planes=438, out_planes=6)
        self.dense4 = DenseBlock(in_planes=6, nb_layers=12, growth_rate=24)
        self.dilation = Dilation()

        # upwards
        self.convT1 = nn.ConvTranspose2d(4096, 512, kernel_size=2, stride=2)
        # self.convT2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)


        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                    #   kernel_size=2, stride=2 doubles size of img
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.conv_start = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        print(x.shape)
        x = self.conv_start(x)
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        dense2 = x
        x = self.trans2(x)
        x = self.dense3(x)
        dense3 = x
        x = self.trans3(x)
        x = self.dense4(x) # [-1, 294, 8, 8]
        # TODO check if dilation is correct
        x = self.dilation(x) # concat 4 dilated [-1, 294, 8, 8] =  [-1, 1176, 8, 8]

        print("DDUnet shape:", x.shape)

        # upwards
        x = self.convT1(x) #  [-1, 512, 16, 16]
        x = torch.cat([x, dense3], 1) # [1, 950, 16, 16]


        print("dense3: ", dense3)

        return x
        # skip_connections = []
        #
        # for down in self.downs:
        #     x = down(x)
        #     skip_connections.append(x)
        #     x = self.pool(x)
        #
        # x = self.bottleneck(x)
        # skip_connections = skip_connections[::-1] # reverse list
        #
        # for idx in range(0, len(self.ups), 2):
        #     x = self.ups[idx](x)
        #     skip_connection = skip_connections[idx//2]
        #
        #     # TODO: This has to be checked
        #     if x.shape != skip_connection.shape:
        #         x = TF.resize(x, size=skip_connection.shape[2:])
        #
        #     concat_skip = torch.cat((skip_connection, x), dim=1)
        #     x = self.ups[idx+1](concat_skip) # Run via Double Conv

        # return self.final_conv(x)


def test():
    x = torch.randn((1, 1, 256, 256))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    # print(preds.shape)
    # print(x.shape)
    # print(model)
    # image = torch.rand((1, 572, 572))
    summary(model, (1, 256, 256))
    # assert preds.shape == x.shape

if __name__ == "__main__":
    test()