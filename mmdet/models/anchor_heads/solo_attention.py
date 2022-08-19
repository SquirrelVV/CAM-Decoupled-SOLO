import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn import Module, Conv2d, Parameter, Softmax, Dropout, Embedding, InstanceNorm2d, ReLU, BatchNorm2d, Sequential, MaxPool2d, ConvTranspose2d
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module']


class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim, load_weights=False):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim   # in_dim: 代表特征图数目
        self.gamma = Parameter(torch.zeros(1))   # gamma初始值，设置为0
        self.softmax = Softmax(dim=-1)     # softmax操作之后，在dim这个维度相加等于1
        if not load_weights:
            self.init_weights()
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)   B：batchsize， C: channel
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()     # 得到特征图的尺寸参数，输入特征图为A
        proj_query = x.view(m_batchsize, C, -1)      # 对特征图进行reshape B
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)   # 对特征图进行reshape并转置 B的转置
        energy = torch.bmm(proj_query, proj_key)     # 计算两个tensor的矩阵乘法，注意两个tensor的维度必须为3
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy    #
        attention = self.softmax(energy_new)   # softmax操作后生成CxC的注意力图 X
        proj_value = x.view(m_batchsize, C, -1)      # 对输入特征图进行reshape B

        out = torch.bmm(attention, proj_value)    # 注意力图与reshape后的特征图相乘
        out = out.view(m_batchsize, C, height, width)  # reshape成特征图同样的大小 D

        out = self.gamma*out + x       # A+D=E， E为融合通道信息的特征图
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

