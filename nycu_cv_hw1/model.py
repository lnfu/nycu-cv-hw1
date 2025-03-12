from torch import nn
from torchvision.models import resnet


class Model(nn.Module):
    def __init__(self, backbone: resnet.ResNet, num_classes=100):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        fc_in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(fc_in_features, num_classes)
        # for name, param in self.backbone.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")

    def forward(self, x):
        return self.backbone(x)
