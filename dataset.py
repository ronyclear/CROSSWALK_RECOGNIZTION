import torch
import os
import json
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
import glob
from PIL import Image
import numpy as np
import cv2



class Attribute():
    def __init__(self):
        self.labels = {}
        self.label_id_to_name = {}
        self.label_name_to_id = {}


class RoadSegDataset(Dataset):
    def __init__(self, data_dir, n_classes):
        self.anno_dir = data_dir
        self.n_classes = n_classes
        self.data = []
        self.labels = []
        self.mask_values = [0, 1]
        img_dir = os.path.join(data_dir, "image")
        label_dir = os.path.join(data_dir, "mask")
        self.img_list = glob.glob(img_dir+"/*.jpg")
        self.label_list = glob.glob(label_dir+"/*.png")

    def __len__(self):
        return len(self.img_list)

    def process(self, mask_values, img, scale, is_mask):
        w, h = img.size
        scale_w, scale_h = int(w * scale), int(h * scale)
        assert scale_w > 0 and scale_h > 0, "scale too small"
        img = img.resize((scale_w, scale_h), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        # 转为numpy
        img = np.asarray(img)
        if is_mask:
            mask = np.zeros((scale_h, scale_w), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...] # 增加一个batch维度
            else:
                img = img.transpose((2,0,1))

            if(img > 1).any():
                img = img / 255.0
            return img

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        label = Image.open(self.label_list[idx])
        img = self.process(self.mask_values, img, scale=0.5, is_mask=False)
        mask = self.process(self.mask_values, label, scale=0.5, is_mask=True)

        return {"img": torch.as_tensor(img.copy()).float().contiguous(),
                "mask": torch.as_tensor(mask.copy()).float().contiguous()}


if __name__ == "__main__":
    data_dir = "./data/train"
    tf = transforms.Compose([transforms.Resize((640, 640)),
                             transforms.ToTensor()])
    rd = RoadSegDataset(data_dir, 1)
    res = rd[1]


