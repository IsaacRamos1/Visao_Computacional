import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(MyModel, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True)
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 32, kernel_size=3)
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.bottleneck(x)
        x = self.decoder1(x)
        x = self.encoder3(x)
        return x

