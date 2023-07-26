import torch
import torch.nn as nn
import torch.nn.functional as F


# A 'model' with 2 learnable weights, learn_beta determines if they should be learnable
# they are made learnable with nn.Parameter()
# sx and sq ARE beta
class GeoPoseLoss(nn.Module):
    def __init__(self, sx=0.0, sq=0.0, learn_beta=False):
        super(GeoPoseLoss, self).__init__()
        self.learn_beta = learn_beta

        if not self.learn_beta:
            self.sx = 0
            self.sq = -6.25

        # the scaling factors sx and sq are defined as trainable parameters(nn.Parameter)
        # and initialized with the provided values(both 0)
        # The requires_grad argument determines whether the parameters will be updated during training
        self.sx = nn.Parameter(torch.Tensor([sx]), requires_grad=self.learn_beta)
        self.sq = nn.Parameter(torch.Tensor([sq]), requires_grad=self.learn_beta)

        self.loss_print = []  # can be used to store all losses

    def forward(self, pred_x, pred_q, target_x, target_q):
        pred_q = F.normalize(pred_q, p=2, dim=1)  # normalise to unit quaternion

        # separate L1 loss for pos and ori, then combine
        loss_x = F.l1_loss(pred_x, target_x)  # MAE; element-wise absolute error, then take mean
        loss_q = F.l1_loss(pred_q, target_q)
        # l2 loss is MSE or Euclidean; element-wise squared error, then take mean


        # overall loss is computed as a combination of the pos and ori losses,
        # weighted by the scaling factors sx and sq, respectively.
        # The scaling factors are used to control the importance of each loss term.
        loss = torch.exp(-self.sx) * loss_x + self.sx \
            + torch.exp(-self.sq) * loss_q + self.sq

        # self.loss_print = [loss.item(), loss_x.item(), loss_q.item()]
        return loss, loss_x.item(), loss_q.item()


if __name__ == '__main__':
    loss_fn = GeoPoseLoss()

    # Create synthetic data (also works for batches)
    pred_x = torch.randn(1, 3)  # (32, 3) etc.
    pred_q = torch.randn(1, 4)
    target_x = torch.randn(1, 3)
    target_q = torch.randn(1, 4)

    loss, loss_x, loss_q = loss_fn(pred_x, pred_q, target_x, target_q)

    print(loss)
    print("Total Loss:", loss.item())
    # PyTorch tensors represent numerical values with additional information
    # such as gradients for automatic differentiation.
    # The.item() method is used to extract the numerical value from a tensor as a Python scalar.
    print("Translation Loss:", loss_x)
    print("Rotation Loss:", loss_q)
