import torch
import torch.nn.functional as F
import torch.nn as nn

def structure_loss(pred, mask):

    # BCE loss
    k = nn.Softmax2d()
    weit = torch.abs(pred-mask)
    weit = k(weit)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3))/weit.sum(dim=(2, 3))

    # IOU loss
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1-(inter+1)/(union-inter+1)

    return (wbce + wiou).mean()