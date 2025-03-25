import logging
import humanize
import torch
import torchvision
import torchinfo


class Model(torch.nn.Module):
    def __init__(self, backbone: torchvision.models.resnet.ResNet, num_classes=100):
        super().__init__()
        self.backbone = backbone
        fc_in_features = self.backbone.fc.in_features

        for param in self.backbone.parameters():
            param.requires_grad = True

        self.backbone.fc = torch.nn.Sequential(
            torch.nn.Linear(fc_in_features, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes),
        )

        num_params = sum(p.numel() for p in self.backbone.parameters())
        logging.info(f"# of parameters = {humanize.intword(num_params)}")
        assert num_params <= 1e8

    def forward(self, x):
        return self.backbone(x)
