#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
from torch.utils.data import Dataset
import h5py

def wgn(x, snr):
    batch_size, len_x = x.shape
    Ps = np.sum(np.power(x, 2)) / len_x
    Pn = Ps / (np.power(10, snr / 10))
    noise = np.random.randn(len_x) * np.sqrt(Pn)
    return x + noise

def load_h5(h5_path,data_aug = False):
    # load training data  读取数据并处理
    with h5py.File(h5_path, 'r') as hf:
        head = list(hf.keys())
        # print('List of arrays in input file:', hf.keys())
        X = np.transpose(np.array(hf.get(head[0]), dtype=np.float32))
        Y = np.transpose(np.array(hf.get(head[1]), dtype=np.float32))
        if X.ndim == 3:
            X1 = X.swapaxes(1, 2)
            Y1 = Y
        elif Y.ndim == 3:
            X1 = Y.swapaxes(1, 2)
            Y1 = X
        else:
            raise RuntimeError("维度错误")
        # 加入数据增强试试！！
        # if data_aug:
        #     number = X1.shape[0]
        #     SNR = [20,10,5]
        #     X2 = np.zeros(number*(len(SNR)+1),X1.shape[1],X1.shape[2])
        #     for i in range(number):
        #         data_I = X1[i, 0, :]
        #         data_Q = X1[i, 1, :]
        #         data_complex = data_I+1j*data_Q


        print("数据维度:",X1.shape)
        print("标签维度:",Y1.shape)
    return X1, Y1


class SignalDataset(Dataset):
    """数据加载器"""
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.X, self.Y = load_h5(data_folder)

    def __getitem__(self, item):
        X = self.X[item]
        Y = self.Y[item]
        return X, Y

    def __len__(self):
        return len(self.Y)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    X, Y = load_h5('Task_1_Test.mat')
    data_I = X[1, 0, :]
    data_Q = X[1, 1, :]
    data_complex = data_I + 1j * data_Q
    print(data_complex.shape)
    print(type(data_complex))
    fig = plt.figure(dpi=150)
    plt.plot(X[190,1,:],label="signal_"+str(int(Y[190][0])))
    plt.legend()
    plt.show()