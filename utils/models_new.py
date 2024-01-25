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
        B = int((1+events[-1,-1]).item())
        num_voxels = int(2 * np.prod(self.dim) * B)
        vox = events[0].new_full([num_voxels,], fill_value=0)
        C, H, W = self.dim

        # get values for each channel
        x, y, t, p, b = events.t()

        # normalizing timestamps
        for bi in range(B):
            t[events[:,-1] == bi] /= t[events[:,-1] == bi].max()

        p = (p+1)/2  # maps polarity to 0, 1

        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \
                          + W * H * C * 2 * b

        for i_bin in range(C):
            values = t * self.value_layer.forward(t-i_bin/(C-1))

            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(-1, 2, C, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)

        return vox


class Classifier(nn.Module):
    def __init__(self,
                 voxel_dimension=(9,180,240),  # dimension of voxel will be C x 2 x H x W
                 crop_dimension=(224, 224),  # dimension of crop before it goes into classifier
                 num_classes=101,
                 mlp_layers=[1, 30, 30, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 pretrained=True):
        nn.Module.__init__(self)
        self.quantization_layer = QuantizationLayer(voxel_dimension, mlp_layers, activation)
        # self.classifier = resnet34(pretrained=pretrained)
        self.classifier = SuperPointNet(num_classes)

        self.crop_dimension = crop_dimension

        # replace fc layer and first convolutional layer
        # input_channels = 2*voxel_dimension[0]
        # self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)

        input_channels = 2*voxel_dimension[0]
        self.classifier.conv1a = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)


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
        vox = self.quantization_layer.forward(x)
        vox_cropped = self.crop_and_resize_to_resolution(vox, self.crop_dimension)
        pred = self.classifier.forward(vox_cropped)
        return pred, vox





class SuperPointNet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self,num_classes):
    super(SuperPointNet, self).__init__()

    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    self.unpool = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)

    ##测试
    self.num_classes = num_classes

    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    det_h = 65
    gn = 64
    useGn = False
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

    ##新增全连接层测试
    self.fc = torch.nn.Linear(c4*28*28,num_classes)
    self.flat = torch.nn.Flatten()

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
        x = self.conv4b(x)
        x = self.bn4b(x)
        x = self.relu(x)
        # x = self.relu(self.bn4b(self.conv4b(x)))
        
        ##测试,输出101
        x = self.flat(x)
        x = self.fc(x) 

        return x
    #     # Detector Head.
    #     cPa = self.relu(self.bnPa(self.convPa(x)))
    #     semi = self.bnPb(self.convPb(cPa))
    #     # Descriptor Head.
    #     cDa = self.relu(self.bnDa(self.convDa(x)))
    #     desc = self.bnDb(self.convDb(cDa))

    # dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    # desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    # output = {'semi': semi, 'desc': desc}

    # return output