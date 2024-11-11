import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(MyModel, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
                                                            ## revisar e consertar erros para MyModel
                                                            ## conflito de tipos de variável crit(output, label)
    def forward(self, x):
        x = self.encoder1(x)
        x = self.flatten(x)
        output = self.classifier(x)
        return output
    

class MyPreTrainedModel(nn.Module):
    def __init__(self, out_channels=5):
        super(MyPreTrainedModel, self).__init__()
        self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, out_channels)

    def forward(self, x):
        return self.model(x)

