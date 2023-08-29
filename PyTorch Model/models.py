import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# https://pytorch.org/vision/stable/models.html
from torchsummary import summary


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
    # seems to be no random crop, no colour jitter;
    # log of quaternion
    # 300 epochs


class ResNet(nn.Module):
    def __init__(self, fixed_weight=False, dropout_rate=0.0):
        super(ResNet, self).__init__()
        base_model = models.resnet34(weights="IMAGENET1K_V1")
        # resnet50(weights="IMAGENET1K_V2")
        self.dropout_rate = dropout_rate
        feat_in = base_model.fc.in_features

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])  # removing final linear layer

        # previously initted with ImageNet weights
        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.fc_last = nn.Linear(feat_in, 2048, bias=True)
        self.fc_position = nn.Linear(2048, 3, bias=True)
        self.fc_rotation = nn.Linear(2048, 4, bias=True)

        # init with kaiming weights
        init_modules = [self.fc_last, self.fc_position, self.fc_rotation]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # weights from a normal distribution and biases to 0
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.base_model(x)
        # print("AFTER RESNET", x.shape)
        x = x.view(x.size(0), -1)  # flatten TODO global average pool
        # print("AFTER VIEW", x.shape)
        x = self.fc_last(x)
        x = F.relu(x)

        dropout_on = self.training
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=dropout_on)

        # from that same space of 2048, 3 for pos and 4 for ori are sampled
        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation


class GoogleNet(nn.Module):
    """ PoseNet using Inception V3 """

    def __init__(self, fixed_weight=False, dropout_rate=0.0):
        super(GoogleNet, self).__init__()
        base_model = models.inception_v3(weights="IMAGENET1K_V1")  # from torchvision models
        self.dropout_rate = dropout_rate

        model = []
        model.append(base_model.Conv2d_1a_3x3)
        model.append(base_model.Conv2d_2a_3x3)
        model.append(base_model.Conv2d_2b_3x3)
        model.append(nn.MaxPool2d(kernel_size=3, stride=2))
        model.append(base_model.Conv2d_3b_1x1)
        model.append(base_model.Conv2d_4a_3x3)
        model.append(nn.MaxPool2d(kernel_size=3, stride=2))
        model.append(base_model.Mixed_5b)
        model.append(base_model.Mixed_5c)
        model.append(base_model.Mixed_5d)
        model.append(base_model.Mixed_6a)
        model.append(base_model.Mixed_6b)
        model.append(base_model.Mixed_6c)
        model.append(base_model.Mixed_6d)
        model.append(base_model.Mixed_6e)
        model.append(base_model.Mixed_7a)
        model.append(base_model.Mixed_7b)
        model.append(base_model.Mixed_7c)
        print(model)
        # Finally,
        self.base_model = nn.Sequential(*model)

        # freezing layers on ImageNet trained weights
        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Out 2 linear layers
        self.pos2 = nn.Linear(2048, 3, bias=True)
        self.ori2 = nn.Linear(2048, 4, bias=True)

    def forward(self, x):
        # 299 x 299 x 3 as this is inception v3
        x = self.base_model(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # training of nn.Module is set to False in eval mode, here we set it explicitly
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        pos = self.pos2(x)
        ori = self.ori2(x)

        return pos, ori


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = GoogleNet()
    # model.to(device)
    # summary(model, (3, 299, 299))

    model2 = PoseNet()
    model2.to(device)
    summary(model2, (3, 224, 224))
