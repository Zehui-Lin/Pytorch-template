import os
import numpy as np
from matplotlib import pyplot as plt
import torch


class AvgMeter(object):
    """ this class is to record one variable such as loss or acc """

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def numpy_to_torch(img, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v


def torch_to_numpy(tensor):
    img = tensor[0].cpu().data.numpy()
    img = img.transpose((1, 2, 0))
    return img

def check_dir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except:
            os.makedirs(path)

def check_empty_dir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except:
            os.makedirs(path)
    else:
        if not os.listdir(path)==[]:
            for i in os.listdir(path):
                os.remove(os.path.join(path, i))

def plot(*args, x_label, y_label, title, save_path):
    '''
    输入数据格式：数据1、图例名1, 数据2、图例名2,......
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    assert len(args) % 2 == 0
    for i in range(len(args)//2):
        ax.plot(args[2*i], label=args[2*i+1])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(save_path, "{}.png".format(title)))
    plt.close()


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


class savebest_weights():
    def __init__(self, num, weight_path):
        super(savebest_weights, self).__init__()
        #  num  表示要保存的模型数量
        #  weight_path表示存储路径
        self.weight_path = weight_path
        self.num = num
        self.acc_list = []
        self.path_list = []
        self.index = 0
        check_empty_dir(self.weight_path)

    def remove(self, file_path):
        if os.path.exists(file_path):  # 如果文件存在
            os.remove(file_path)
        else:
            pass

    def save_weight(self, net, acc, epoch):
        path = os.path.join(self.weight_path, "Network_{}_{:.4f}.pth.gz".format(epoch, acc))

        if len(self.acc_list) < self.num:  # 如果保存模型数量小于预设值，继续保存
            self.acc_list.append(acc)
            self.path_list.append(path)
            state = net.state_dict()
            torch.save(state, path)
        else:  # 如果大于等于预设值，按acc大小排序
            self.acc_array = np.array(self.acc_list)
            self.index = np.argsort(-self.acc_array, axis=0)
            self.index = self.index[:self.num]
            self.acc = self.acc_array[self.index]

            if acc >= self.acc[-1]:  # 如果新的acc大于之前的最小值，则保存模型，同时删除acc最小的那个模型
                self.remove(self.path_list[self.index[-1]])
                del self.acc_list[self.index[-1]]
                del self.path_list[self.index[-1]]
                self.acc_list.append(acc)
                self.path_list.append(path)
                state = net.state_dict()
                torch.save(state, path)
            else:
                pass