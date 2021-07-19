import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils import is_tracing

from ..builder import HEADS
from .cls_head import ClsHead


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


@HEADS.register_module()
class ArcFaceHead(ClsHead):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel. 
                       if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """
    def __init__(self,
                 num_classes,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 s = 64.0, 
                 m = 0.50):
        super(ArcFaceHead, self).__init__(init_cfg=init_cfg)
        self.in_features = in_channels
        self.out_features = num_classes

        self.s = s
        self.m = m

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self._init_layers()

    def _init_layers(self):
        self.kernel = nn.Parameter(torch.FloatTensor(self.in_features, self.out_features))
        #nn.init.xavier_uniform_(self.kernel)
        nn.init.normal_(self.kernel, std=0.01)

    def forward_train(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis = 1)
        kernel_norm = l2_norm(self.kernel, axis = 0)
        cos_theta = torch.mm(embbedings, kernel_norm).clamp(-1, 1)  # for numerical stability
        # with torch.no_grad():
        #     origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        final_target_logit = torch.where(target_logit > self.th, cos_theta_m, target_logit - self.mm)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        losses = self.loss(output, label)
        return losses

    def simple_test(self, embbedings):
        """Test without augmentation."""

        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm).clamp(-1, 1)  # for numerical stability
        cls_score = cos_theta * self.s
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None

        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

