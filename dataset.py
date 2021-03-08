'''
Author: Zehui Lin
Date: 2021-01-22 17:20:36
LastEditTime: 2021-03-08 12:16:14
LastEditors: Zehui Lin
Description: 
'''
import os
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
from imgaug import augmenters as iaa
from torch.utils.data import Dataset


class MySet(Dataset):
    def __init__(self, txt_path, mode="train", is_debug=False):
        self.mode = mode
        self.data_buffer = read_buffer(self.mode, txt_path, is_debug)

    def __getitem__(self, item):
        data_item = self.data_buffer[item]
        raw_data = data_item["image"]
        label = data_item["label"]
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Crop(percent=(0, 0.1)),
            iaa.Multiply((0.8, 1.2))],
            random_order=True)
        seq_same = seq.to_deterministic()

        if self.mode == "train":
            # cv2.imwrite('./data/original/'+str(item)+'.png', raw_data)  # save original image
            processed_data = seq_same(image=raw_data)
            # cv2.imwrite('./data/augmentation/'+str(item)+'.png', processed_data)  # visualize the augmentation effect
        else:
            processed_data = raw_data

        data = torch.from_numpy(processed_data.transpose((2, 0, 1)).astype(np.float32)/255)

        return data, label

    def __len__(self):
        return len(self.data_buffer)


def read_buffer(mode, txt_path, is_debug=False):
    if mode == "train":
        txt_read_path = os.path.join(txt_path, "train.txt")
    elif mode == "val":
        txt_read_path = os.path.join(txt_path, "val.txt")
    elif mode == "test":
        txt_read_path = os.path.join(txt_path, "test.txt")
    elif mode == "all":
        txt_read_path = os.path.join(txt_path, "all.txt")
    else:
        raise ValueError

    fid = open(txt_read_path, "r")
    lines = fid.readlines()
    random.shuffle(lines) # avoid all same label in smoke test
    if is_debug:
        tiny_set = lines[0:int(1/10*len(lines))]
        lines = tiny_set
    fid.close()

    data_buffer = []
    for line in tqdm(lines, desc="Loading: "+mode+" data"):
        line = line.strip()
        data_path, label = line.split(",") 

        raw_data = cv2.imread(data_path)

        label = torch.from_numpy(np.array(label, dtype=np.int64))

        image_buffer = {
            "image": raw_data,
            "label": label
        }

        data_buffer.append(image_buffer)

    return data_buffer
