import torch
import torch.nn as nn
import math

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_dropout=False, use_bn=False):
        super(ResidualBlock, self).__init__()
        layers = [
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        ]

        if use_bn:
            layers.append(nn.BatchNorm2d(channels))  # Insert BN after first conv
        layers.append(nn.PReLU())

        if use_dropout:
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)  # Residual connection


class Generator(nn.Module):
    def __init__(self, num_res_blocks=12, scale_factor=4):
        super(Generator, self).__init__()
        self.scale_factor = scale_factor
        assert math.log2(scale_factor).is_integer(), "Scale factor must be a power of 2"

        # Initial conv
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(64, use_dropout=True, use_bn=True) for _ in range(num_res_blocks)])

        # Mid conv (no activation to match original SRGAN)
        self.mid_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Upsampling
        self.upsample = self.make_upsample_layers(scale_factor)

        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4),
            nn.Tanh()  # Output in range [-1, 1]
        )

        self._initialize_weights()

    def make_upsample_layers(self, scale_factor):
        layers = []
        num_blocks = int(math.log2(scale_factor))
        for _ in range(num_blocks):
            layers += [
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            ]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out1 = self.initial(x)
        out2 = self.res_blocks(out1)
        out3 = self.mid_conv(out2)
        out3 += out1  # Global skip connection
        out4 = self.upsample(out3)
        return self.final(out4)