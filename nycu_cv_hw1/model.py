import logging
import humanize
import torch
import torchvision


class Model(torch.nn.Module):
    def __init__(self, backbone: torchvision.models.resnet.ResNet, num_classes=100):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        fc_in_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(fc_in_features, num_classes)
        # for name, param in self.backbone.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")
        num_params = sum(p.numel() for p in self.backbone.parameters())
        logging.info(f"# of parameters = {humanize.intword(num_params)}")

    def forward(self, x):
        return self.backbone(x)
