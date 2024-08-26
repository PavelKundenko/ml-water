import torch
import torch.nn as nn
import torch.nn.functional as F


# U-Net Model
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder (Downsampling Path)
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder (Upsampling Path)
        self.decoder4 = self.upconv_block(1024, 512)
        self.decoder3 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder1 = self.upconv_block(128, 64)

        # Output layer
        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        # A block of 2 convolutional layers with Batch Normalization and ReLU activation
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        # A block for upsampling the feature map followed by a convolutional block
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self.conv_block(out_channels, out_channels)
        )

    def forward(self, x):
        # Encoding path
        e1 = self.encoder1(x)  # First encoder layer
        e2 = self.encoder2(F.max_pool2d(e1, kernel_size=2, stride=2))  # Second encoder layer
        e3 = self.encoder3(F.max_pool2d(e2, kernel_size=2, stride=2))  # Third encoder layer
        e4 = self.encoder4(F.max_pool2d(e3, kernel_size=2, stride=2))  # Fourth encoder layer

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, kernel_size=2, stride=2))  # Bottleneck layer

        # Decoding path
        d4 = self.decoder4(b)  # First decoder layer
        d3 = self.decoder3(d4 + F.interpolate(e4, d4.size()[2:], mode='bilinear', align_corners=True))  # Second decoder layer with skip connection
        d2 = self.decoder2(d3 + F.interpolate(e3, d3.size()[2:], mode='bilinear', align_corners=True))  # Third decoder layer with skip connection
        d1 = self.decoder1(d2 + F.interpolate(e2, d2.size()[2:], mode='bilinear', align_corners=True))  # Fourth decoder layer with skip connection

        # Output layer
        output = self.output_layer(d1 + F.interpolate(e1, d1.size()[2:], mode='bilinear', align_corners=True))  # Output layer with skip connection
        return torch.sigmoid(output)  # Sigmoid activation for binary classification
