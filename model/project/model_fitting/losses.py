import torch.nn.functional as F
import torch

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, label):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = label * distances + (1 + -1 * label) * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2)
        return losses