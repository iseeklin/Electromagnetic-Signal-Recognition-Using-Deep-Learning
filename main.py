#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# TODO python main.py --model ConvNetVGG --eval-freq 1 --save-dir ConvNetVGG_log/
# python main.py --model ConvNet --eval-freq 1 --save-dir ConvNet_log/
# python main.py --model complexCNN2 --eval-freq 1 --save-dir complexCNN2_log/
import torch.nn.functional as F
import os
import sys
import argparse
import datetime
import time
import os.path as osp
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from pytorchtools import EarlyStopping
import mydata_read
import mymodel1
from utils import AverageMeter, Logger
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser("ADS-B signal recognition Example")
# dataset
parser.add_argument('--class_num', type=int, default=100)
parser.add_argument('-j', '--workers', default=0, type=int,
                     help="number of data loading workers (default: 0)")
# optimization
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--lr-model', type=float, default=0.005, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=150, help="max train epoch")  # 100
parser.add_argument('--stepsize', type=int, default=15)# 学习率的调整步长
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")  # 学习率衰减
# model
parser.add_argument('--model', type=str, default='ConvNet')
#
parser.add_argument('--eval-freq', type=int, default=1)# Test数据进行验证的频率
parser.add_argument('--print-freq', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--data_aug', action='store_true')
parser.add_argument('--save-dir', type=str, default='mylog')

args = parser.parse_args()

def main():
    filepath = 'dataNormCut.mat'
    patience = 13
    print('data path:',filepath)
    path1 = osp.join(args.save_dir + args.model + ".pt")
    print("model path:", path1)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
    sys.stdout = Logger(osp.join(args.save_dir, 'log' +'.txt'))
    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.manual_seed(args.seed)  # 设置随机种子
    else:
        print("####Currently using CPU")
    print("Creating dataset: ")
    data_set = mydata_read.SignalDataset(filepath)#DT_V8X400X5
    length = len(data_set)
    print('data_set length:',length)
    train_size, validate_size,test_size = int(0.8 * length), int(0.1 * length), length-int(0.9 * length)
    train_set, validate_set, test_set = torch.utils.data.random_split(data_set, [train_size, validate_size,test_size])
    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(validate_set, batch_size=args.batch_size, shuffle=True)

    print("Creating model: {}".format(args.model))

    model = mymodel1.create(name=args.model, num_classes=args.class_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if use_gpu:
        model = nn.DataParallel(model).cuda()# 多GPU训练
    model = model.cuda()

    criterion_xent = nn.CrossEntropyLoss()
    optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr_model)

    if args.stepsize > 0:
       scheduler = lr_scheduler.ExponentialLR(optimizer_model, 0.9, last_epoch=-1)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path1)
    acc_train = []
    acc_eval = []
    loss_train = []
    loss_eval = []
    start_time = time.time()
    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))  # {}  打印格式化
        acctrain, losstrain= train(model, criterion_xent, optimizer_model,
              trainloader, use_gpu, args.class_num, epoch)
        loss_train.append(losstrain)
        acc_train.append(acctrain)
        if args.stepsize > 0 and (epoch + 1) % 1 == 0:  scheduler.step()
        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            acc, losseval = eval(model, valloader, use_gpu, criterion_xent, args.class_num, epoch)
            print("Train_Accuracy (%): {:.4f}\tEval_Accuracy (%): {:.4f}  ".format(acctrain, acc))
            print("Train_Loss (%): {:.4f}\tEval_Loss (%): {:.4f}  ".format(losstrain, losseval))
            acc_eval.append(acc)
            loss_eval.append(losseval)
            early_stopping(losseval, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    fig = plt.figure(1)
    plt.plot(acc_train, label="train_acc")
    plt.plot(acc_eval, label="eval_acc")
    plt.legend()
    plt.show()
    fig = plt.figure(2)
    plt.plot(loss_train, label="train_loss")
    plt.plot(loss_eval, label="eval_loss")
    plt.legend()
    plt.show()
    print("#######开始测试")
    model.load_state_dict(torch.load(path1))
    print('导入模型成功')
    testloder = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
    real_label, pre_label = test(model, testloder, use_gpu)
    reallabel = []
    predlabel = []
    for i in range(len(real_label)):
        reallabel += real_label[i]
        predlabel += pre_label[i]
    import Confusion_matrix
    Confusion_matrix.plot_confusion_matrix(reallabel, predlabel, args.save_dir ,cmap=plt.cm.Blues)

def train(model, criterion_xent, optimizer_model, trainloader,use_gpu, num_classes, epoch):
    model.train()
    losses = AverageMeter()
    correct, total = 0, 0
    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        outputs = model(data)
        preds = F.log_softmax(outputs, dim=1)
        labels = torch.squeeze(labels)
        loss_xent = criterion_xent(preds, labels.long())
        loss = loss_xent
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()
        losses.update(loss.item(), labels.size(0))
        total += len(preds)
        correct += (preds.argmax(dim=1) == labels).sum().item()

    acc = correct * 100. / total
    return acc, losses.avg


def eval(model, testloader, use_gpu, criterion_xent, num_classes, epoch):
    model.eval()
    losses = AverageMeter()
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            labels = torch.squeeze(labels)
            outputs = model(data)
            preds = F.log_softmax(outputs, dim=1)
            eval_loss = criterion_xent(preds, labels.long())
            losses.update(eval_loss.item(), labels.size(0))
            total += len(preds)
            correct += (preds.argmax(dim=1) == labels).sum().item()
    acc = correct * 100. / total
    return acc, losses.avg


def test(model, testloader, use_gpu):
    model.eval()
    real_label, pre_label =[], []
    feature = []
    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            labels = torch.squeeze(labels)
            outputs = model(data)
            predictions = outputs.data.max(1)[1]
            real_label.append(labels.data.cpu().numpy().tolist())
            pre_label.append(predictions.cpu().numpy().tolist())

    return real_label, pre_label

if __name__ == '__main__':
    main()

