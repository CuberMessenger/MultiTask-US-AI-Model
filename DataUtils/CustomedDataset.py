import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from skimage import exposure
from torch.utils.data import Dataset
from StatisticsUtils import SaveTensorImage
from DataUtils.Preprocess import CenterResize

UltrasoundDataTransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees = 0, translate = (0.05, 0.05), scale = (0.95, 1.05)),
    transforms.ToTensor()])

class TensorDatasetWithTransform(Dataset):
    def __init__(self, tensors, dataFolderPath, dataIndexes, transform = None):
        self.Tensors = tensors
        self.Transform = transform
        self.DataFolderPath = dataFolderPath
        self.DataIndexes = dataIndexes

    def __getitem__(self, index):
        dataIndex = self.DataIndexes[index]#from 0, 1, 2, 3 to 45, 47, 4675, 3654...
        image = cv2.imread(os.path.join(self.DataFolderPath, dataIndex + "_Image.jpg"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.DataFolderPath, dataIndex + "_Mask.jpg"), cv2.IMREAD_GRAYSCALE)
        image, mask = CenterResize(image, mask, expandRate = 1.2)
        image = cv2.resize(image, (256, 256))
        image = exposure.equalize_adapthist(image, clip_limit = 0.015) * 255

        image = torch.Tensor(image) / 255

        label = self.Tensors[0][index]

        if self.Transform is not None:
            image = self.Transform(image)
        else:
            image = image[None, :, :]

        return image, label, dataIndex

    def __len__(self):
        return self.Tensors[0].size(0)