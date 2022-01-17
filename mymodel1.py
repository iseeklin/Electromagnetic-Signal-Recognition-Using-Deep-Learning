#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import complexnn
# 有全连接层
class ConvNet(nn.Module):
    def __init__(self, num_cls=8):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 2 * 25, kernel_size=11, stride=1, padding=5, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(2 * 25, 3 * 25, kernel_size=11, stride=1, padding=5, bias=True),
            nn.BatchNorm1d(3 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            # nn.Dropout(0.25),

            nn.Conv1d(3 * 25, 4 * 25, kernel_size=11, stride=1, padding=5, bias=True),
            nn.BatchNorm1d(4 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            # nn.Dropout(0.25),

            nn.Conv1d(4 * 25, 6 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(6 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            # nn.Dropout(0.25),

            nn.Conv1d(6 * 25, 8 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(8 * 25),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            # nn.Dropout(0.25),

            nn.Conv1d(8 * 25, 12 * 25, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm1d(12 * 25),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            # nn.Dropout(0.25)# 表示舍去节点的个数
        )
        self.fc1 = nn.Linear(300*75, 256)#
        self.prelu_fc1 = nn.PReLU()
        self.Dropout =nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_cls)  # 没有进行归一化，转化为独热编码


    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = x.view(-1,300*75)
        x = self.prelu_fc1(self.fc1(x))
        x = self.Dropout(x)
        y = self.fc3(x)
        return y

# 没有全连接层
class ConvNet1(nn.Module):
    def __init__(self, num_cls=8):
        super(ConvNet1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 2 * 25, kernel_size=11, stride=1, padding=5, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(2 * 25, 3 * 25, kernel_size=11, stride=1, padding=5, bias=True),
            nn.BatchNorm1d(3 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            # nn.Dropout(0.25),

            nn.Conv1d(3 * 25, 4 * 25, kernel_size=11, stride=1, padding=5, bias=True),
            nn.BatchNorm1d(4 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            # nn.Dropout(0.25),

            nn.Conv1d(4 * 25, 6 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(6 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            # nn.Dropout(0.25),

            nn.Conv1d(6 * 25, 8 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(8 * 25),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            # nn.Dropout(0.25),

            nn.Conv1d(8 * 25, 12 * 25, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm1d(12 * 25),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            # nn.Dropout(0.25)# 表示舍去节点的个数
        )
        self.fc1 = nn.Linear(300*37, num_cls)#

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = x.view(-1,300*37)
        y = self.fc1(x)
        return y

class ConvNetVGG(nn.Module):
    def __init__(self, num_cls=30):
        super(ConvNetVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, stride=4),

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, stride=4),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm1d(256),
            # nn.ReLU(inplace=True),
            nn.MaxPool1d(4, stride=4),

            nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=1, bias=True),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512*23, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_cls)
        )
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        # print(x.shape)
        x = x.view(-1,512*23)
        y = self.classifier(x)
        return y

class complexCNN2(nn.Module):
    def __init__(self, num_class):
        super(complexCNN2, self).__init__()
        self.features = nn.Sequential(
            complexnn.ComplexConv(1, 32, kernel_size=11, stride=1, padding=5, bias=True),
            nn.BatchNorm1d(32*2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            complexnn.ComplexConv(32, 64, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),

            complexnn.ComplexConv(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            complexnn.ComplexConv(128, 128, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),


            complexnn.ComplexConv(128, 256, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            complexnn.ComplexConv(256, 256, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)

        )
        self.fc1 = nn.Linear(512 * 25, 256)  #
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, num_class)  # 没有进行归一化，转化为独热编码

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = x.view(-1, 512 * 25)
        x = self.prelu_fc1(self.fc1(x))
        x = self.fc2(x)
        y = self.fc4(x)
        return y


__factory = {
    'ConvNet': ConvNet,# 实数CNN BN+droup
    'ConvNet1': ConvNet1,# 实数CNN 仅仅BN
    'ConvNetVGG': ConvNetVGG,
    'complexCNN2': complexCNN2,# 复数CNN

}

def create(name, num_classes):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](num_classes)

if __name__ == '__main__':
    pass
