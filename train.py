import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
import cv2
from torch import nn
from models.AlexNet import AlexNet
from data.MyDataset import BitmojiDataset
from torch.utils.tensorboard import SummaryWriter

from models.VGGNet import  vgg


def train(net, train_loader, num_epochs, loss, optimizer, num_batchs):
    writer = SummaryWriter("runs/VGG")
    net.train()
    for i in range(num_epochs):
        for index, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_hat = net(x)
            l = loss(y_hat, y.long())
            l.backward()
            optimizer.step()
            train_loss = l
            train_acc = accuracy(y_hat, y)
            writer.add_scalar("loss", scalar_value=train_loss, global_step=i * num_batchs + index)
            writer.add_scalar("accuracy", scalar_value=train_acc, global_step=i * num_batchs + index)
            print(f'epoch: {i}, batch: {index}, loss: {train_loss:.3f}, train acc {train_acc:.3f}')

    writer.close()
    torch.save(net.state_dict(), 'wights/VGG.params')


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = (y_hat == y).long()
    return torch.sum(cmp) / len(y)


if __name__ == '__main__':
    train_img_dir = "D:/data/dl/BitmojiDataset/trainimages/"
    filename = "D:/data/dl/train.csv"

    num_epochs = 5
    batch_size = 256
    lr = 0.0001

    conv_arch = ((1, 32), (1, 64), (1, 128), (1, 256), (1, 256))
    net = vgg(conv_arch)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train_dataset = BitmojiDataset(filename=filename, image_dir=train_img_dir, resize_height=224,
                                   resize_width=224)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    num_batchs = len(train_dataset) // batch_size + 1

    train(net, train_loader, num_epochs, loss, optimizer, num_batchs)
