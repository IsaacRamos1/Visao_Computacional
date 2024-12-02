import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=5):
        super(MyModel, self).__init__()
        self._out_channels = out_channels

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=1)    # outra opção avgpooling2D
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.encoder1(x)
        x = self.flatten(x)
        
        self.dense = nn.Linear(x.shape[1], self._out_channels).to('cuda:0')
        x = self.dense(x)
        return x
    

class MyPreTrainedModel(nn.Module):
    def __init__(self, name: str, out_channels=5):
        super(MyPreTrainedModel, self).__init__()
        self._activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        #self._activation = nn.Softmax(dim=-1)

        self._name = name
        #self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        #self.model.fc = nn.Linear(self.model.fc.in_features, out_channels)

        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, out_channels)
        
        #self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)   
        #self.model.fc = nn.Linear(self.model.fc.in_features, out_channels)
        #self.model.aux_logits = False

        #self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        #in_features = self.model.classifier[-1].in_features
        #self.model.classifier[-1] = nn.Linear(in_features, out_channels)

    def forward(self, x):
        #return self._activation(self.model(x))
        return self.model(x)

