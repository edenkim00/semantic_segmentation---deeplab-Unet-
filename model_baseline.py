import torch
import torch.nn as nn
import torch.nn.functional as F

class segnet(nn.Module):
    def __init__(self, out_channel=10):
        super(segnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(16, out_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.conv4(out)
        return out

class aspp(nn.Module):
    def __init__(self, in_channel, out_channel,dilation=1):
        super(aspp, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.conv_1_1 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.conv_3_1 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=dilation*1, dilation=dilation*1)
        self.conv_3_2 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=dilation*2, dilation=dilation*2)
        self.conv_3_3 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=dilation*3, dilation=dilation*3)
        self.conv_1_2 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.BatchNorm = nn.BatchNorm2d(out_channel)
        self.conv_output = nn.Conv2d(out_channel * 5, out_channel, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        size = x.shape[2:]
        out1 = self.conv_1_1(x)
        out1 = self.BatchNorm(out1)
        out1 = self.relu(out1)
        out2 = self.conv_3_1(x)
        out2 = self.BatchNorm(out2)
        out2 = self.relu(out2)
        out3 = self.conv_3_2(x)
        out3 = self.BatchNorm(out3)
        out3 = self.relu(out3)
        out4 = self.conv_3_3(x)
        out4 = self.BatchNorm(out4)
        out4 = self.relu(out4)
        out5 = self.avgpool(x)
        out5 = self.conv_1_2(out5)
        out5 = self.BatchNorm(out5)
        out5 = self.relu(out5)
        out5 = F.interpolate(out5, size=size, mode = "bilinear")
        out = torch.cat([out1, out2, out3,out4,out5],1)
        out = self.conv_output(out)
        return out

class UNetDec(nn.Module):

    def __init__(self, in_channels, features, out_channels, dilation=1):
        super().__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, 1, padding = dilation, dilation = dilation),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, 1, padding = dilation, dilation = dilation),
            nn.BatchNorm2d(features),
        )
        if in_channels != features:
            conv = nn.Conv2d(in_channels, features, kernel_size=1, stride=1, bias=False)
            bn = nn.BatchNorm2d(features)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()
        self.up2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features, out_channels, 2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.up(x)
        out = out + self.downsample(x)
        out = self.up2(out)
        return out


class UNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, dilation=1):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, 1, padding = dilation, dilation = dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, padding = dilation, dilation = dilation),
            nn.BatchNorm2d(out_channels),
        ]
        self.down = nn.Sequential(*layers)
        self.relu = nn.ReLU()
        if dropout:
            self.maxpool_dropout = nn.Sequential(nn.Dropout(.5), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        else :
            self.maxpool_dropout = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        if in_channels != out_channels:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        out = self.down(x)
        out = out + self.downsample(x)
        out = self.relu(out)
        out = self.maxpool_dropout(out)
        return out



class UNet(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.enc0 = UNetEnc(3, 32, dilation = 1)
        self.enc1 = UNetEnc(32, 64, dilation = 1)
        self.enc2 = UNetEnc(64, 128, dilation = 1)
        self.enc3 = UNetEnc(128, 256, dilation = 2)
        self.enc4 = UNetEnc(256, 512, dilation = 2, dropout=True)

        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, 3,1,1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3,1,1),
            nn.BatchNorm2d(1024),
        )
        # self.center_residual = nn.Conv2d(512, 1024, kernel_size=1,stride = 1, padding = 0)
        self.after_center = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.ReLU(inplace=True),
        )
        # self.Aspp1 = aspp(64,64,4)
        # self.Aspp2 = aspp(256,256,6)
        self.dec4 = UNetDec(1024, 512, 256, dilation = 1)
        self.dec3 = UNetDec(512, 256, 128, dilation = 1)
        self.dec2 = UNetDec(256, 128, 64, dilation = 1)
        self.dec1 = UNetDec(128, 64, 32, dilation = 1)
        self.dec0 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        center = self.center(enc4)
        # center_res = self.center_residual(enc4)
        # center += center_res
        center = self.after_center(center)

        dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, size = center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, size = dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, size = dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, size = dec2.size()[2:], mode='bilinear')], 1))
        dec0 = self.dec0(torch.cat([dec1, F.interpolate(enc0, size = dec1.size()[2:], mode='bilinear')], 1))

        return F.interpolate(self.final(dec0), size = x.size()[2:], mode= 'bilinear')


if __name__ == "__main__":
    batch = torch.zeros(64, 3, 256, 256)
    model = segnet()
    output = model(batch)
    print(output.size())