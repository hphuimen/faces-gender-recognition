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
from torch.nn import functional as F

from models.VGGNet import vgg
from utils import *

def test(net, test_loader):
    net.load_state_dict(torch.load("wights/VGG.params"))
    net.eval()
    labels = []
    y_pre = []
    for index, (x, y) in enumerate(test_loader):
        y_hat = net(x)
        y_hat = F.softmax(y_hat, dim=1).detach().numpy()[0]
        with torch.no_grad():
            y_hat = y_hat.argmax()
            labels.append(int(y.numpy()[0]))
            y_pre.append(int(y_hat))

    return labels, y_pre






if __name__ == '__main__':
    conv_arch = ((1, 32), (1, 64), (1, 128), (1, 256), (1, 256))
    net = vgg(conv_arch)
    filename = "D:/data/dl/test.csv"

    test_img_dir = "D:/data/dl/BitmojiDataset/testimages/"

    test_dataset = BitmojiDataset(filename=filename,image_dir=test_img_dir, resize_height=224,resize_width=224,
                                  is_training=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    labels, y_pre =test(net, test_loader)

    acc = accuracy_test(labels, y_pre)
    precision = precision_test(labels, y_pre)
    recall = recall_test(labels, y_pre)
    f1 = f1_test(labels, y_pre)
    print("acc: %f, precision: %f, recall: %f ,f1: %f" %(acc, precision, recall, f1))

    class_names = np.array(["female", "male"])
    plot_confusion_matrix(labels, y_pre, class_names, title="VGG confusion matrix")
    plt.show()

    plot_roc(labels, y_pre, title="VGG ROC Curve")







