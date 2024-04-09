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

class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")
        # path = False
        if isfile(path):
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None,...,None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in tqdm.tqdm(range(1000)):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()


    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels-1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels-1)] = 0
        gt_values[ts > 1.0 / (num_channels-1)] = 0

        return gt_values

class QuantizationLayer(nn.Module):
    def __init__(self, dim,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1)):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers,
                                      activation=activation,
                                      num_channels=dim[0])
        self.dim = dim

    def forward(self, events):
        # points is a list, since events can have any size
        B = int((1+events[-1,-1]).item()) #这里的B是一个loader里面装载的“图片”数量
        num_voxels = int(2 * np.prod(self.dim) * B)
        vox = events[0].new_full([num_voxels,], fill_value=0)
        T, H, W = self.dim

        # get values for each channel
        x, y, t, p, b = events.t()

        # normalizing timestamps
        for bi in range(B):
            t[events[:,-1] == bi] /= t[events[:,-1] == bi].max()

        p = (p+1)/2  # maps polarity to 0, 1

        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * T * p \
                          + W * H * T * 2 * b #将位置，极性，通道，loader的装载量(batch_size)转换到编码
        
        for i_bin in range(T):
            values = t * self.value_layer.forward(t-i_bin/(T-1)) #t已经归一化了，就是将一个事件流分成8段

            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin #idx是加了8个通道过后的位置编码
            vox.put_(idx.long(), values, accumulate=True) #put_方法就是按一维排序的方法，将values,也就是value_layers编码后的时间信息，放到vox中

        vox = vox.view(-1, 2, T, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1) #把两个9通道合在一起，变成18通道的vox

        return vox

#仿照long_lived的结构
# 定义Squeeze-and-Excite ResNet块
class SEResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SEResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.se(out) * out
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out,residual

# 定义整个网络
class CornerHeatmap(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CornerHeatmap, self).__init__()
        self.se_resnet1 = SEResNetBlock(in_channels, 64)
        self.se_resnet2 = SEResNetBlock(64, 128)
        self.conv_lstm1 = ConvLSTM(128, 128, (3,3), 2)
        self.se_resnet3 = SEResNetBlock(128, 128)
        self.se_resnet4 = SEResNetBlock(128, 64)
        self.conv_lstm2 = ConvLSTM(64, 64, (3,3), 2)
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        out,res = self.se_resnet1(x)
        skip_connections.append(res)

        out,res = self.se_resnet2(out+skip_connections[-1])
        skip_connections.append(res)

        out = self.conv_lstm1(out)

        out,res = self.se_resnet3(out+skip_connections[-1])
        skip_connections.append(res)

        out,res = self.se_resnet4(out+skip_connections[-1])
        skip_connections.append(res)

        out = self.conv_lstm2(out)

        out = self.conv_out(out+skip_connections[-1])
        return out

#参照stable keypoint,使用角点heatmap
class EventCornerHeatmap(nn.Module):
    def __init__(self,
                 voxel_dimension=(10,260,346),  # dimension of voxel will be C x 2 x H x W，生成数据集是仿DAVIS346,即（260,246）
                 crop_dimension=(224, 224),  # dimension of crop before it goes into classifier
                 num_classes=2,
                 mlp_layers=[1, 30, 30, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1)
                 ):
        nn.Module.__init__(self)
        self.quantization_layer = QuantizationLayer(voxel_dimension, mlp_layers, activation)
        self.crop_dimension = crop_dimension
        self.backbone = CornerHeatmap(voxel_dimension[0],voxel_dimension[0])
    
    # def crop_and_resize_to_resolution(self, x, output_resolution=(224, 224)):
    #     B, C, H, W = x.shape
    #     if H > W:
    #         h = H // 2
    #         x = x[:, :, h - W // 2:h + W // 2, :]
    #     else:
    #         h = W // 2
    #         x = x[:, :, :, h - H // 2:h + H // 2]

    #     x = F.interpolate(x, size=output_resolution)

    #     return x

    def forward(self, x):
        # vox = self.quantization_layer.forward(x)

        # 输入就是固定的vox
        vox = x
        # vox_cropped = self.crop_and_resize_to_resolution(vox, self.crop_dimension)
        pred = self.backbone.forward(vox)

        return pred, vox

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