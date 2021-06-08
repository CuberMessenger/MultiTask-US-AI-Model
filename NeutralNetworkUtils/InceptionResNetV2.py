import torch
import torch.nn as nn

class BasicConvolution2D(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize, stride, padding = 0):
        super().__init__()
        self.Convolution = nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding, bias = False)
        self.BatchNormalization = nn.BatchNorm2d(outChannels, eps = 0.001, momentum = 0.1, affine = True)
        self.ReLU = nn.ReLU(inplace = False)

    def forward(self, x):
        x = self.Convolution(x)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        return x

class Mixed5B(nn.Module):
    def __init__(self):
        super().__init__()
        self.Branch0 = BasicConvolution2D(192, 96, kernelSize = 1, stride = 1)
        self.Branch1 = nn.Sequential(
            BasicConvolution2D(192, 48, kernelSize = 1, stride = 1),
            BasicConvolution2D(48, 64, kernelSize = 5, stride = 1, padding = 2))
        self.Branch2 = nn.Sequential(
            BasicConvolution2D(192, 64, kernelSize = 1, stride = 1),
            BasicConvolution2D(64, 96, kernelSize = 3, stride = 1, padding = 1),
            BasicConvolution2D(96, 96, kernelSize = 3, stride = 1, padding = 1))
        self.Branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride = 1, padding = 1, count_include_pad = False),
            BasicConvolution2D(192, 64, kernelSize = 1, stride =1))

    def forward(self, x):
        x0 = self.Branch0(x)
        x1 = self.Branch1(x)
        x2 = self.Branch2(x)
        x3 = self.Branch3(x)
        x = torch.cat((x0, x1, x2, x3), 1)
        return x

class Block35(nn.Module):
    def __init__(self, scale = 1.0):
        super().__init__()
        self.Scale = scale
        self.Branch0 = BasicConvolution2D(320, 32, kernelSize = 1, stride = 1)
        self.Branch1 = nn.Sequential(
            BasicConvolution2D(320, 32, kernelSize = 1, stride = 1),
            BasicConvolution2D(32, 32, kernelSize = 3, stride = 1, padding = 1))
        self.Branch2 = nn.Sequential(
            BasicConvolution2D(320, 32, kernelSize = 1, stride = 1),
            BasicConvolution2D(32, 48, kernelSize = 3, stride = 1, padding = 1),
            BasicConvolution2D(48, 64, kernelSize = 3, stride = 1, padding = 1))
        self.Convolution = nn.Conv2d(128, 320, kernel_size = 1, stride = 1)
        self.ReLU = nn.ReLU(inplace = False)

    def forward(self, x):
        x0 = self.Branch0(x)
        x1 = self.Branch1(x)
        x2 = self.Branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.Convolution(out)
        out = out * self.Scale + x
        out = self.ReLU(out)
        return out

class Mixed6A(nn.Module):
    def __init__(self):
        super().__init__()
        self.Branch0 = BasicConvolution2D(320, 384, kernelSize = 3, stride = 2)
        self.Branch1 = nn.Sequential(
            BasicConvolution2D(320, 256, kernelSize = 1, stride = 1),
            BasicConvolution2D(256, 256, kernelSize = 3, stride = 1, padding = 1),
            BasicConvolution2D(256, 384, kernelSize = 3, stride = 2))
        self.Branch2 = nn.MaxPool2d(3, stride = 2)

    def forward(self, x):
        x0 = self.Branch0(x)
        x1 = self.Branch1(x)
        x2 = self.Branch2(x)
        x = torch.cat((x0, x1, x2), 1)
        return x

class Block17(nn.Module):
    def __init__(self, scale = 1.0):
        super().__init__()
        self.Scale = scale
        self.Branch0 = BasicConvolution2D(1088, 192, kernelSize = 1, stride = 1)
        self.Branch1 = nn.Sequential(
            BasicConvolution2D(1088, 128, kernelSize = 1, stride = 1),
            BasicConvolution2D(128, 160, kernelSize = (1, 7), stride = 1, padding = (0, 3)),
            BasicConvolution2D(160, 192, kernelSize = (7, 1), stride = 1, padding = (3, 0)))
        self.Convolution = nn.Conv2d(384, 1088, kernel_size = 1, stride = 1)
        self.ReLU = nn.ReLU(inplace = False)

    def forward(self, x):
        x0 = self.Branch0(x)
        x1 = self.Branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.Convolution(out)
        out = out * self.Scale + x
        out = self.ReLU(out)
        return out

class Mixed7A(nn.Module):
    def __init__(self):
        super().__init__()
        self.Branch0 = nn.Sequential(
            BasicConvolution2D(1088, 256, kernelSize = 1, stride = 1),
            BasicConvolution2D(256, 384, kernelSize = 3, stride = 2))
        self.Branch1 = nn.Sequential(
            BasicConvolution2D(1088, 256, kernelSize = 1, stride = 1),
            BasicConvolution2D(256, 288, kernelSize = 3, stride = 2))
        self.Branch2 = nn.Sequential(
            BasicConvolution2D(1088, 256, kernelSize = 1, stride = 1),
            BasicConvolution2D(256, 288, kernelSize = 3, stride = 1, padding = 1),
            BasicConvolution2D(288, 320, kernelSize = 3, stride = 2))
        self.Branch3 = nn.MaxPool2d(3, stride = 2)

    def forward(self, x):
        x0 = self.Branch0(x)
        x1 = self.Branch1(x)
        x2 = self.Branch2(x)
        x3 = self.Branch3(x)
        x = torch.cat((x0, x1, x2, x3), 1)
        return x

class Block8(nn.Module):
    def __init__(self, scale = 1.0, needReLU = True):
        super().__init__()
        self.Scale = scale
        self.NeedReLU = needReLU
        self.Branch0 = BasicConvolution2D(2080, 192, kernelSize = 1, stride = 1)
        self.Branch1 = nn.Sequential(
            BasicConvolution2D(2080, 192, kernelSize = 1, stride = 1),
            BasicConvolution2D(192, 224, kernelSize = (1, 3), stride = 1, padding = (0, 1)),
            BasicConvolution2D(224, 256, kernelSize = (3, 1), stride = 1, padding = (1, 0)))
        self.Convolution = nn.Conv2d(448, 2080, kernel_size = 1, stride = 1)
        if self.NeedReLU:
            self.ReLU = nn.ReLU(inplace = False)

    def forward(self, x):
        x0 = self.Branch0(x)
        x1 = self.Branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.Convolution(out)
        out = out * self.Scale + x
        if self.NeedReLU:
            out = self.ReLU(out)
        return out

class MultiScaleBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride, padding = 0):
        super(MultiScaleBlock, self).__init__()
        self.Branch0 = BasicConvolution2D(inChannel, outChannel // 4, kernelSize = 3, stride = stride, padding = 0 + padding)
        self.Branch1 = BasicConvolution2D(inChannel, outChannel // 4, kernelSize = 5, stride = stride, padding = 1 + padding)
        self.Branch2 = BasicConvolution2D(inChannel, outChannel // 4, kernelSize = 7, stride = stride, padding = 2 + padding)
        self.Branch3 = BasicConvolution2D(inChannel, outChannel // 4, kernelSize = 9, stride = stride, padding = 3 + padding)

    def forward(self, x):
        x0 = self.Branch0(x)
        x1 = self.Branch1(x)
        x2 = self.Branch2(x)
        x3 = self.Branch3(x)
        return torch.cat((x0, x1, x2, x3), 1)

class InceptionResNetV2(nn.Module):
    def __init__(self, numOfClasses):
        super().__init__()
        self.Convolution1A = BasicConvolution2D(1, 32, kernelSize = 3, stride = 2)
        self.Convolution2A = BasicConvolution2D(32, 32, kernelSize = 3, stride = 1)
        self.Convolution2B = BasicConvolution2D(32, 64, kernelSize = 3, stride = 1, padding = 1)
        self.MaxPooling3A = nn.MaxPool2d(3, stride = 2)
        self.Convolution3B = BasicConvolution2D(64, 80, kernelSize = 1, stride = 1)
        self.Convolution4A = BasicConvolution2D(80, 192, kernelSize = 1, stride = 1)
        self.MaxPooling5A = nn.MaxPool2d(3, stride = 2)
        self.Mixed5B = Mixed5B()
        self.Repeat0 = nn.Sequential(
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17))
        self.Mixed6A = Mixed6A()
        self.Repeat1 = nn.Sequential(
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10))
        self.Mixed7A = Mixed7A()
        self.Repeat2 = nn.Sequential(
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20))
        self.Block8 = Block8(needReLU = False)
        self.Convolution7B = BasicConvolution2D(2080, 1536, kernelSize = 1, stride = 1)
        self.AveragePooling1A = nn.AvgPool2d(6, count_include_pad = False)
        self.LastLinear = nn.Linear(1536, numOfClasses)
    
    def forward(self, x):
        # 1 256 256 512
        x = self.Convolution1A(x)# 32 127 127 255
        x = self.Convolution2A(x)# 32 125 125 253
        x = self.Convolution2B(x)# 64 125 125 253
        x = self.MaxPooling3A(x)# 64 62 62 126
        x = self.Convolution3B(x)# 80 62 62 126
        x = self.Convolution4A(x)# 192 62 62 126
        x = self.MaxPooling5A(x)# 192 30 30 62
        x = self.Mixed5B(x)# 320 30 30 62
        x = self.Repeat0(x)# 320 30 30 62
        x = self.Mixed6A(x)# 1088 14 14 30
        x = self.Repeat1(x)# 1088 14 14 30
        x = self.Mixed7A(x)# 2080 6 6 14
        x = self.Repeat2(x)# 2080 6 6 14
        x = self.Block8(x)# 2080 6 6 14
        x = self.Convolution7B(x)# 1536 6 6 14
        x = self.AveragePooling1A(x)
        x = x.view(x.size(0), -1)
        x = self.LastLinear(x)
        return x