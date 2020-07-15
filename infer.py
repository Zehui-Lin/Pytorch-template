from network import model
from dataset import MySet
from torch.utils.data import DataLoader
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def test(net, loader):
    net.eval()
    predicts, labels = [], []
    for batch_idx, (data, label) in enumerate(loader):  # 遍历
        data = data.cuda()
        y = net(data)
        predict = y.data.cpu().numpy()
        label = label.data.numpy()
        predicts.extend(np.argmax(predict, axis=1))
        labels.extend(label)
    acc = accuracy_score(labels, predicts)
    tn, fp, fn, tp = confusion_matrix(labels, predicts).ravel()
    return acc, tn, fp, fn, tp


name = ''
txt_path = 'data/txt'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# 模型
output_path = "output_test/"+name
best_path = os.path.join(output_path, "best_weights")
best_weight = os.path.join(best_path, '15_0_2_0.9074.pth.gz')
net = model().cuda()
net.load_state_dict(torch.load(best_weight))
# 数据
test_set = MySet(txt_path, mode="test")
test_loader = DataLoader(test_set, batch_size=10, num_workers=20)
# 测试
test_acc, tn, fp, fn, tp = test(net, test_loader)
precision = tp/(tp+fp)
sensitive = tp/(tp+fn)
specificity = tn/(tn+fp)
F1score = 2*tp/(2*tp+fp+fn)
print("The test acc:{}".format(test_acc))
print("The precision:{}|sensitive:{}|specificity:{}|F1score:{}".format(precision, sensitive, specificity, F1score)