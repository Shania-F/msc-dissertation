import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# https://github.com/NVlabs/geomapnet/blob/master/models/posenet.py
class PoseNet(nn.Module):
    def __init__(self, droprate=0.5, pretrained=True):
        super(PoseNet, self).__init__()
        self.droprate = droprate

        # replace the last FC layer in feature extractor
        self.feature_extractor = models.resnet34(weights="IMAGENET1K_V1")
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        in_dim = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(in_dim, 2048)

        self.fc_xyz = nn.Linear(2048, 3)
        self.fc_wpqr = nn.Linear(2048, 4)

        # initialize
        if pretrained:
            init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        # 1 x 2048
        x = F.relu(x)
        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return xyz, wpqr
