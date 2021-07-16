import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
import torch

def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    """Sigmoid focal loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, *).
        target (torch.Tensor): The ground truth label of the prediction with
            shape (N, *).
        weight (torch.Tensor, optional): Sample-wise loss weight with shape
            (N, ). Dafaults to None.
        gamma (float): The gamma for calculating the modulating factor.
            Defaults to 2.0.
        alpha (float): A balanced form for Focal Loss. Defaults to 0.25.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' ,
            loss is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: Loss.
    """
    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class FocalLoss(nn.Module):
    """Focal loss.

    Args:
        gamma (float): Focusing parameter in focal loss.
            Defaults to 2.0.
        alpha (float): The parameter in balanced form of focal
            loss. Defaults to 0.25.
        reduction (str): The method used to reduce the loss into
            a scalar. Options are "none" and "mean". Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):

        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Sigmoid focal loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, *).
            target (torch.Tensor): The ground truth label of the prediction
                with shape (N, *).
            weight (torch.Tensor, optional): Sample-wise loss weight with shape
            (N, *). Dafaults to None.
            avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss into a scalar. Options are "none", "mean" and "sum".
                Defaults to None.

        Returns:
            torch.Tensor: Loss.
        """
        pred=torch.reshape(pred,target.shape)
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * sigmoid_focal_loss(
            pred,
            target,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_cls

# # todo 集成多分类的focalloss
# class FocalLoss(nn.Module):
#     def __init__(self, opt):
#         super(FocalLoss, self).__init__()
#         self.weights = opt.LOSS.FocalLoss.weights
#         self.gamma = opt.LOSS.FocalLoss.gamma
#         cls_weights = opt.LOSS.FocalLoss.cls_weights
#         assert len(cls_weights) == opt.DATASET.NUM_CLASSES
#         self.cls_weights = torch.Tensor(cls_weights).cuda()
#
#
#     def forward(self, inputs, labels, **kwargs):
#         inputs = inputs["logits"]
#         inputs = to_list(inputs)
#
#         losses = []
#         for input in inputs:
#             """
#             cal culates loss
#             logits: batch_size * seq_length
#             labels: batch_size
#             """
#
#             # transpose labels into labels onehot
#             new_label = labels.unsqueeze(1)
#             label_onehot = torch.zeros([input.shape[0], input.shape[1]]).cuda().scatter_(1, new_label, 1)
#             alpha = label_onehot * self.cls_weights.expand(input.shape[0], input.shape[1])
#
#             # calculate log
#             log_p = F.log_softmax(input, dim=-1)
#             pt = alpha * log_p
#             sub_pt = 1 - pt
#             fl = -(sub_pt) ** self.gamma * log_p
#             losses.append(fl.mean())
#         return sum(losses) / len(losses) * self.weights
