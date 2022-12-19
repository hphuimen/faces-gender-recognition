import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
import cv2
from PIL import Image


class BitmojiDataset(Dataset):
    def __init__(self, filename, image_dir, resize_height=None, resize_width=None, transform=None, is_training=True):
        """
        :param filename: 数据文件csv：格式：image_name.jpg label
        :param image_dir: 图片路径：image_dir+image_name.jpg构成图片的完整路径
        :param resize_height 为None时，不进行缩放
        :param resize_width  为None时，不进行缩放，
        """


        self.file_name, self.image_label = self.read_file(filename)

        self.image_dir = image_dir
        self.len = len(self.file_name)
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.transform = transform
        self.is_training = is_training

    def __getitem__(self, index):

        label = self.image_label[index]
        image_name = self.file_name[index]
        image_path = os.path.join(self.image_dir, image_name)
        img = self.load_data(image_path)
        img = self.data_preproccess(img, self.transform,
                                    self.is_training, self.resize_height, self.resize_width)
        if label<0:
            label=0


        return img, label

    def __len__(self):
        return self.len

    def read_file(self, filename):
        data = pd.read_csv(filename).T.values
        return data[0], data[1].astype(float)

    def read_testfile(self):
        test_files = []
        for i in range(3000, 4084):
            test_file = str(i) + ".jpg"
            test_files.append(test_file)
        return np.array(test_files)

    def load_data(self, path):
        """
        加载数据
        :param path:
        :return:
        """
        bgr_image = cv2.imread(path)

        if bgr_image is None:
            print("Warning:不存在:{}", path)
            return None
        if len(bgr_image.shape) == 2:  # 若是灰度图则转为三通道
            print("Warning:gray image", path)
            bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        return rgb_image

    def data_preproccess(self, data, transform, is_training, resize_height, resize_width):
        """
        数据预处理
        :param resize_width:
        :param resize_height:
        :param is_training:
        :param transform:
        :param data:
        :return:
        """
        if resize_height is None:
            resize_height = data.shape[0]
        if resize_width is None:
            resize_width = data.shape[1]

        data = Image.fromarray(data)

        if transform is not None:
            data = transform(data)
        elif is_training:
            normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            train_transforms = transforms.Compose([
                transforms.Resize((resize_height, resize_width)),
                transforms.ToTensor(),
                normalize
            ])
            data = train_transforms(data)
        else:
            normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            test_transforms = transforms.Compose([
                transforms.Resize((resize_height, resize_width)),
                transforms.ToTensor(),
                normalize
            ])
            data = test_transforms(data)

        return data


if __name__ == '__main__':
    train_img_dir = "D:/data/dl/BitmojiDataset/trainimages/"
    test_img_dir = "D:/data/dl/BitmojiDataset/testimages/"
    filename = "D:/data/dl/test.csv"
    '''epoch = 1
    batch_size = 128

    train_dataset = BitmojiDataset(filename=filename, image_dir=train_img_dir)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    for i in range(epoch):
        for index, (images, labels) in enumerate(train_loader):
            print(images.shape, labels.shape, images, labels)
            break'''
    test_dataset = BitmojiDataset(filename=filename, image_dir=test_img_dir,is_training=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

    for i in range(1):
        for index, (images, labels) in enumerate(test_loader):
            print(images.shape, labels.shape, images, labels)
            break
