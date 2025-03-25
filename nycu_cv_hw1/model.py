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
        for layer in self.backbone.children():
            for bottleneck in layer.children():
                for l in bottleneck.children():
                    if l.__class__.__name__ == "BatchNorm2d":
                        for param in l.parameters():
                            param.requires_grad = True
                    else:
                        for param in l.parameters():
                            param.requires_grad = False

        self.backbone.fc = torch.nn.Linear(fc_in_features, num_classes)
        torch.nn.init.xavier_uniform_(self.backbone.fc.weight)

        num_params = sum(p.numel() for p in self.backbone.parameters())
        logging.info(f"# of parameters = {humanize.intword(num_params)}")
        assert num_params <= 1e8

        # self.backbone.fc = torch.nn.Sequential(
        #     torch.nn.Linear(fc_in_features, 1024),
        #     torch.nn.GELU(),
        #     torch.nn.Linear(1024, num_classes),  # 最終輸出層
        # )
        # for param in self.backbone.parameters():
        #     param.requires_grad = True
        # for param in self.backbone.layer4.parameters():
        #     param.requires_grad = True
        # for param in self.backbone.fc.parameters():
        #     param.requires_grad = True

        # for name, param in self.backbone.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")

    def forward(self, x):
        return self.backbone(x)
