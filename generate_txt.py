import os
import torch
import argparse
import numpy as np
from utils import check_dir
from dataset import ImageSet
from network import Full_Share
from network import Partial_Share
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
from numpy import random as random
from utils import check_dir
from init import InitParser

data_path = './'
txtdir_path = './txt'
class1_path = data_path + ''
class2_path = data_path + ''

check_dir(txtdir_path)

class1_list = os.listdir(class1_path)
class2_list = os.listdir(class2_path)

class1_num = len(class1_list)
class2_num = len(class2_list)


train_file = open(os.path.join(txtdir_path, "train.txt"), "w")
val_file = open(os.path.join(txtdir_path, "val.txt"), "w")
test_file = open(os.path.join(txtdir_path, "test.txt"), "w")

# train
for data in class1_list[0:int(0.6*class1_num)]:
    data_name = os.path.join(class1_path, data)
    train_file.write("{} {}\n".format(data_name, 1))

for data in class2_list[0:int(0.6*class2_num)]:
    data_name = os.path.join(class2_path, data)
    train_file.write("{} {}\n".format(data_name, 0))

# val
for data in class1_list[int(0.6*class1_num):int(0.8*class1_num)]:
    data_name = os.path.join(class1_path, data)
    val_file.write("{} {}\n".format(data_name, 1))

for data in class2_list[int(0.6*class2_num):int(0.8*class2_num)]:
    data_name = os.path.join(class2_path, data)
    val_file.write("{} {}\n".format(data_name, 0))

# test
for data in class1_list[int(0.8*class1_num):]:
    data_name = os.path.join(class1_path, data)
    test_file.write("{} {}\n".format(data_name, 1))

for data in class2_list[int(0.8*class2_num):]:
    data_name = os.path.join(class2_path, data)
    test_file.write("{} {}\n".format(data_name, 0))


train_file.close
val_file.close
test_file.close

print("generate  finished!")
