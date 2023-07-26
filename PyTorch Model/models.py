import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# https://pytorch.org/vision/stable/models.html
from torchsummary import summary


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
        x = x.view(x.size(0), -1)  # flatten
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


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))  # cluster centres
        self._init_params()

    def _init_params(self):
        # manually initialising rather than PyTorch assigning random values to weights and 0s to bias
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C = x.shape[:2]  # N->batch size, C->channels

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class ResNetVLAD(nn.Module):
    def __init__(self, fixed_weight=False):
        super(ResNetVLAD, self).__init__()

        # Discard layers at the end of base network
        encoder = models.resnet34(weights="IMAGENET1K_V1")
        self.base_model = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4
        )

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        dim = list(self.base_model.parameters())[-1].shape[0]  # last channels (512)
        self.net_vlad = NetVLAD(num_clusters=32, dim=dim, alpha=1.0)

        self.fc_last = nn.Linear(16384, 2048, bias=True)
        self.fc_position = nn.Linear(2048, 3, bias=True)
        self.fc_rotation = nn.Linear(2048, 4, bias=True)

        init_modules = [self.fc_last, self.fc_position, self.fc_rotation]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # print("FORWARD PASS STARTING", x.shape)
        x = self.base_model(x)
        # print("AFTER RESNET", x.shape)
        embedded_x = self.net_vlad(x)
        # print("AFTER NETVLAD", embedded_x.shape)

        x = self.fc_last(embedded_x)
        x = F.relu(x)

        # from that space of 2048, 3 for pos and 4 for ori are sampled
        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation


if __name__ == '__main__':
    # model = GoogleNet()
    # model.to("cuda:0")
    # summary(model, (3, 299, 299))

    # model2 = ResNet()
    # model2.to("cuda:0")
    # summary(model2, (3, 224, 224))

    model3 = ResNetVLAD()
    model3.to("cuda:0")
    summary(model3, (3, 224, 224))
