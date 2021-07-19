import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils import is_tracing

from ..builder import HEADS
from .cls_head import ClsHead

@HEADS.register_module()
class CircleLoss(ClsHead):
    def __init__(self,
                 in_features,
                 out_features,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 m=0.25,
                 gamma=256):
        super(CircleLoss, self).__init__(init_cfg=init_cfg)
        self.margin = m
        self.gamma = gamma
        self.class_num = out_features
        self.emdsize = in_features

        self.weight = nn.Parameter(torch.FloatTensor(self.class_num, self.emdsize))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        similarity_matrix = nn.functional.linear(nn.functional.normalize(input ,p=2, dim=1, eps=1e-12), nn.functional.normalize(self.weight ,p=2, dim=1, eps=1e-12))

        one_hot = torch.zeros_like(similarity_matrix)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot = one_hot.type(dtype=torch.bool)
        # sp = torch.gather(similarity_matrix, dim=1, index=label.unsqueeze(1))
        sp = similarity_matrix[one_hot]
        mask = one_hot.logical_not()
        sn = similarity_matrix[mask]

        sp = sp.view(input.size()[0], -1)
        sn = sn.view(input.size()[0], -1)

        ap = torch.clamp_min(-sp.detach() + 1 + self.margin, min=0.)
        an = torch.clamp_min(sn.detach() + self.margin, min=0.)

        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        output = torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1)

        losses = self.loss(output, label)
        return losses

    def simple_test(self, input):
        """Test without augmentation."""
        similarity_matrix = nn.functional.linear(nn.functional.normalize(input, p=2, dim=1, eps=1e-12),
                                                 nn.functional.normalize(self.weight, p=2, dim=1, eps=1e-12)).clamp(-1, 1)
        if isinstance(similarity_matrix, list):
            similarity_matrix = sum(similarity_matrix) / float(len(similarity_matrix))
        pred = F.softmax(similarity_matrix, dim=1) if similarity_matrix is not None else None

        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred



@HEADS.register_module()
class CircleLossTrip(ClsHead):
    def __init__(self,
                 m=0.25,
                 gamma=256,
                 similarity='cos'):
        super(CircleLossTrip, self).__init__()
        self.margin = m
        self.gamma = gamma
        self.similarity=similarity

    def forward(self, inputs):
        p = inputs[0]
        n = inputs[1]
        q = inputs[2]

        if self.similarity == 'dot':
            sim_p = self.dot_similarity(q, p)
            sim_n = self.dot_similarity(q, n)
        elif self.similarity == 'cos':
            sim_p = self.cosine_similarity(q, p)
            sim_n = self.cosine_similarity(q, n)
        else:
            raise ValueError('This similarity is not implemented.')

        alpha_p = torch.relu(-sim_p + 1 + self.margin)
        alpha_n = torch.relu(sim_n + self.margin)
        margin_p = 1 - self.margin
        margin_n = -self.margin

        logit_p = torch.reshape(self.gamma * alpha_p * (sim_p - margin_p), (-1, 1))
        logit_n = torch.reshape(self.gamma * alpha_n * (sim_n - margin_n), (-1, 1))

        label_p = torch.ones_like(logit_p)
        label_n = torch.zeros_like(logit_n)
        losses = self.loss(torch.cat([logit_p, logit_n], dim=0),torch.cat([label_p, label_n], dim=0))

        return losses

    def dot_similarity(self, x, y):
        x = torch.reshape(x, (x.shape[0], -1))
        y = torch.reshape(y, (y.shape[0], -1))
        return torch.dot(x, torch.transpose(y,0,1))

    def cosine_similarity(self, x, y):
        x = torch.reshape(x, (x.shape[0], -1))
        y = torch.reshape(y, (y.shape[0], -1))
        abs_x = torch.sqrt(torch.sum(torch.square(x), dim=1, keepdim=True))
        abs_y = torch.sqrt(torch.sum(torch.square(y), dim=1, keepdim=True))
        up = torch.dot(x, torch.transpose(y,0,1))
        down = torch.dot(abs_x, torch.transpose(abs_y,0,1))
        return up / down

    def simple_test(self, input):
        """Test without augmentation."""
        pass
