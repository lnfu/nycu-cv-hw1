from torch import nn
from torchvision.models import resnet


class Model(nn.Module):
    def __init__(self, backbone: resnet.ResNet, num_classes=100):
        super().__init__()
        self.backbone = backbone
        fc_in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(fc_in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
