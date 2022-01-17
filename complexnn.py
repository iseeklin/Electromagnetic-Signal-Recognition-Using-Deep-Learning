import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ComplexConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding = padding
        ## Model components
        self.conv_re = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):  # shpae of x : [batch,2,channel,axis1,axis2]
        n = x.size()[1]
        m = int(n/2)
        x_real = x[:, :m]
        x_imag = x[:, m:]
        real = self.conv_re(x_real) - self.conv_im(x_imag)
        imaginary = self.conv_re(x_real) + self.conv_im(x_imag)
        output = torch.cat((real, imaginary), dim=1)
        return output




        

