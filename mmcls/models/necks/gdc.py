import torch
import torch.nn as nn

from ..builder import NECKS

class Linear_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


@NECKS.register_module()
class GlobalDepthwiseConv(nn.Module):
    def __init__(self, widen_factor, embedding_size=512,kernel=(7,7)):
        super(GlobalDepthwiseConv, self).__init__()
        self.in_channel=512 * widen_factor
        self.conv_6_dw = Linear_block(self.in_channel, self.in_channel, groups=self.in_channel, kernel=kernel, stride=(1, 1), padding=(0, 0))
        self.linear = nn.Conv2d(self.in_channel, out_channels=embedding_size, kernel_size=1, groups=1, stride=1, padding=(0,0), bias=False)  # 映射到embsize，若embsize和gdc出来一致，可省略
        self.bn = nn.BatchNorm2d(embedding_size)  
        self.flatten = nn.Flatten()

    def init_weights(self):
        pass

    def forward(self, inputs):
        
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.conv_6_dw(inputs)
            outs = self.linear(outs)
            outs = self.bn(outs)
            outs = self.flatten(outs)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
