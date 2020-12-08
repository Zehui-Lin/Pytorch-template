import cv2
from imgaug import augmenters as iaa
import numpy as np
import torch
from torch.utils.data import Dataset
import os


class MySet(Dataset):
    def __init__(self, txt_path, mode="train", is_debug=False):
        self.mode = mode
        self.data_buffer = read_buffer(self.mode, txt_path, is_debug)

    def __getitem__(self, item):
        # 数据增强应该放在这里，放在init的函数中没有太大意义，只增强一次
        data_item = self.data_buffer[item]
        raw_data = data_item["image"]
        label = data_item["label"]
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Crop(percent=(0, 0.1)),
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.Sometimes(0.5, iaa.GammaContrast((0.6, 1.67))),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255)),
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )], random_order=True)
        seq_same = seq.to_deterministic()

        if mode == "train":
            # cv2.imwrite('./data/original'+str(item)+'.png', raw_data)  # 保存原图
            processed_data = seq_same(image=raw_data)
            # cv2.imwrite('./data/augmentation'+str(item)+'.png', processed_data)  # 观察增强效果
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
    if is_debug:
        tiny_set = lines[0:int(1/10*len(lines))]
        lines = tiny_set
    fid.close()

    data_buffer = []
    for line in lines:
        line = line.strip()  # 去除末尾的换行符
        data_path, label = line.split(",")  # 指定空格键为分隔符

        # 读图
        raw_data = cv2.imread(data_path)

        label = torch.from_numpy(np.array(label, dtype=np.int64))

        image_buffer = {
            "image": raw_data,
            "label": label
        }

        data_buffer.append(image_buffer)

    return data_buffer
