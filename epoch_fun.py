import numpy as np
import torch
import torch.nn as nn
from utils import AvgMeter
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def train(net, loader, optimizer, cost):
    net.train()
    loss_meter = AvgMeter()
    labels, predicts = [], []
    for batch_idx, (data, label) in tqdm(enumerate(loader)):  # 遍历
        data = data.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        y = net(data)
        loss = cost(y, label)
        loss_meter.update(loss.item())
        loss.backward()
        optimizer.step()
        # 计算acc
        predict = y.data.cpu().numpy()
        label = label.data.cpu().numpy()
        predicts.extend(np.argmax(predict, axis=1))
        labels.extend(label)
        if batch_idx % 10 == 0:
            info = [batch_idx, loss_meter.val]
            print("\rBatch: {} Loss: {:.4f}".format(*info), end="")
    acc = accuracy_score(labels, predicts)
    return loss_meter.avg, acc


def val(net, loader, cost):
    net.eval()
    labels, predicts = [], []
    loss_meter = AvgMeter()
    for batch_idx, (data, label) in tqdm(enumerate(loader)):  # 遍历
        data = data.cuda()
        label = label.cuda()
        y = net(data)
        loss = cost(y, label)
        loss_meter.update(loss.item())
        predict = y.data.cpu().numpy()
        label = label.data.cpu().numpy()
        predicts.extend(np.argmax(predict, axis=1))
        labels.extend(label)
    acc = accuracy_score(labels, predicts)
    return loss_meter.avg, acc


def test(net, loader):
    net.eval()
    labels, predicts = [], []
    for batch_idx, (data, label) in tqdm(enumerate(loader)):  # 遍历
        data = data.cuda()
        y = net(data)
        predict = y.data.cpu().numpy()
        label = label.data.numpy()
        predicts.extend(np.argmax(predict, axis=1))
        labels.extend(label)
    acc = accuracy_score(labels, predicts)
    tn, fp, fn, tp = confusion_matrix(labels, predicts).ravel()
    return acc



