import numpy as np
import torch
import torch.nn as nn
from apex import amp
from tqdm import tqdm
from utils import AvgMeter
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from torch_ema import ExponentialMovingAverage


def train(net, loader, optimizer, cost, apex_training, ema):
    net.train()
    loss_meter = AvgMeter()
    labels, predicts = [], []
    with tqdm(total=len(loader)) as pbar:
        for batch_idx, (data, label) in tqdm(enumerate(loader)):
            data = data.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            y = net(data)
            loss = cost(y, label)
            loss_meter.update(loss.item())
            if apex_training:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model.parameters())
            # 计算acc
            predict = y.data.cpu().numpy()
            label = label.data.cpu().numpy()
            predicts.extend(np.argmax(predict, axis=1))
            labels.extend(label)
            pbar.update(1)
    acc = accuracy_score(labels, predicts)
    return loss_meter.avg, acc


def val(net, loader, cost):
    net.eval()
    labels, predicts = [], []
    loss_meter = AvgMeter()
    for batch_idx, (data, label) in tqdm(enumerate(loader)):
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
    for batch_idx, (data, label) in tqdm(enumerate(loader)):
        data = data.cuda()
        y = net(data)
        predict = y.data.cpu().numpy()
        label = label.data.numpy()
        predicts.extend(np.argmax(predict, axis=1))
        labels.extend(label)
    acc = accuracy_score(labels, predicts)
    tn, fp, fn, tp = confusion_matrix(labels, predicts).ravel()
    return acc, tn, fp, fn, tp
