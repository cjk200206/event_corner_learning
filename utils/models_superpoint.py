"""
    从superpoint迁移网络结构
"""
import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
import tqdm
from .utils.convlstm import ConvLSTM

# 仿照superpoint的结构
# 迁移来的Superpoint
class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self):
        super(SuperPointNet, self).__init__()

        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)

        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65
        gn = 64
        # useGn = False
        useGn = True
        self.reBn = True
        if self.reBn:
            print ("model structure: relu - bn - conv")
        else:
            print ("model structure: bn - relu - conv")


        if useGn:
            print ("apply group norm!")
        else:
            print ("apply batch norm!")


        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.bn1a = nn.GroupNorm(gn, c1) if useGn else nn.BatchNorm2d(c1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.GroupNorm(gn, c1) if useGn else nn.BatchNorm2d(c1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.GroupNorm(gn, c2) if useGn else nn.BatchNorm2d(c2)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.GroupNorm(gn, c2) if useGn else nn.BatchNorm2d(c2)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.bn3a = nn.GroupNorm(gn, c3) if useGn else nn.BatchNorm2d(c3)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.GroupNorm(gn, c3) if useGn else nn.BatchNorm2d(c3)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.bn4a = nn.GroupNorm(gn, c4) if useGn else nn.BatchNorm2d(c4)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.bn4b = nn.GroupNorm(gn, c4) if useGn else nn.BatchNorm2d(c4)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.GroupNorm(gn, c5) if useGn else nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.GroupNorm(det_h, det_h) if useGn else nn.BatchNorm2d(65)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.GroupNorm(gn, c5) if useGn else nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.GroupNorm(gn, d1) if useGn else nn.BatchNorm2d(d1)
        # subpixel head
        # self.predict_flow4 = predict_flow(c4)
        # self.predict_flow3 = predict_flow(c3 + 2)
        # self.predict_flow2 = predict_flow(c2 + 2)
        # self.predict_flow1 = predict_flow(c1 + 2)
        # self.conv_up3 = convrelu(512 + 1024, 512, 3, 1)
        # self.conv_up2 = convrelu(512 + 512, 512, 3, 1)
        # self.conv_up1 = convrelu(256 + 512, 256, 3, 1)
        # self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

    def forward(self, x, subpixel=False):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
        x: Image pytorch tensor shaped N x 1 x H x W.
        Output
        semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
        desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """

        # Let's stick to this version: first BN, then relu
        if self.reBn:
            # Shared Encoder.
            x = self.relu(self.bn1a(self.conv1a(x)))
            conv1 = self.relu(self.bn1b(self.conv1b(x)))
            x, ind1 = self.pool(conv1)
            x = self.relu(self.bn2a(self.conv2a(x)))
            conv2 = self.relu(self.bn2b(self.conv2b(x)))
            x, ind2 = self.pool(conv2)
            x = self.relu(self.bn3a(self.conv3a(x)))
            conv3 = self.relu(self.bn3b(self.conv3b(x)))
            x, ind3 = self.pool(conv3)
            x = self.relu(self.bn4a(self.conv4a(x)))
            x = self.relu(self.bn4b(self.conv4b(x)))
            

            # Detector Head.
            cPa = self.relu(self.bnPa(self.convPa(x)))
            semi = self.bnPb(self.convPb(cPa))
            # desc = 0 ##仅使用semi
            # Descriptor Head.
            cDa = self.relu(self.bnDa(self.convDa(x)))
            desc = self.bnDb(self.convDb(cDa))

            dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
            desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
            # output = {'semi': semi, 'desc': desc}
            # output = semi
        return semi,desc

#参照eventpoint,使用superpoint类
class EventCornerSuperpoint(nn.Module):
    def __init__(self,
                 voxel_dimension=(10,260,346),  # dimension of voxel will be C x 2 x H x W，生成数据集是仿DAVIS346,即（260,246）
                 crop_dimension=(224, 224),  # dimension of crop before it goes into classifier
                 num_classes=2,
                 mlp_layers=[1, 30, 30, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1)
                 ):
        nn.Module.__init__(self)
        self.backbone = SuperPointNet()
        self.crop_dimension = crop_dimension
    
    def crop_and_resize_to_resolution(self, x, output_resolution=(224, 224)):
        B, C, H, W = x.shape
        if H > W:
            h = H // 2
            x = x[:, :, h - W // 2:h + W // 2, :]
        else:
            h = W // 2
            x = x[:, :, :, h - H // 2:h + H // 2]

        x = F.interpolate(x, size=output_resolution)

        return x


    def forward(self, x):
        # vox = self.quantization_layer.forward(x)

        # 输入就是固定的vox
        vox = x
        vox_cropped = self.crop_and_resize_to_resolution(vox, self.crop_dimension)
        semi,desc = self.backbone.forward(vox_cropped)

        return semi, desc

# 作为网络局部测试
if __name__ == '__main__':
    # # 创建网络实例
    # input_channels = 3
    # output_channels = 1
    # network = MyNetwork(input_channels, output_channels)

    # # 打印网络结构
    # print(network)

    # reverse_value_layer = ValueLayer_reverse(mlp_layers=[1, 100, 100, 1])
    print("finish initialization")