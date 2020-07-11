from network import model
from dataset import MySet
from torch.utils.data import DataLoader
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score


def test(net, loader):
    net.eval()
    predicts, labels = [], []
    for batch_idx, (data, label) in enumerate(loader):  # 遍历
        data = data.cuda()
        y = net(data)
        predict = y.data.cpu().numpy()
        label = label.data.numpy()
        predicts.extend(predict)
        labels.extend(label)
    acc = accuracy_score(labels, predicts)
    return acc


name = ''
txt_path = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# 模型
output_path = ""+name
best_path = os.path.join(output_path, "model_op_weight")
best_weight = os.path.join(best_path, '15_0_2_0.9074.pth.gz')
net = model().cuda()
net.load_state_dict(torch.load(best_weight)['net'])
# 数据
test_set = MySet(txt_path, mode="test")
test_loader = DataLoader(test_set, batch_size=10, num_workers=20)
# 测试
test_acc = test(net, test_loader)
print("The test acc:{}".format(test_acc))
