import torch.nn.functional as F
import torch.nn.init

from .common import *
from .convlstm import *
# from models.template import Template
# from utils.losses import *


class FPNEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=512, recurrent=False):
        super(FPNEncoder, self).__init__()

        self.conv_bottom_0 = ConvBlock(
            in_channels=in_channels,
            out_channels=32,
            n_convs=2,
            kernel_size=1,
            padding=0,
            downsample=False,
        )
        self.conv_bottom_1 = ConvBlock(
            in_channels=32,
            out_channels=64,
            n_convs=2,
            kernel_size=5,
            padding=0,
            downsample=False,
        )
        self.conv_bottom_2 = ConvBlock(
            in_channels=64,
            out_channels=128,
            n_convs=2,
            kernel_size=5,
            padding=0,
            downsample=False,
        )
        self.conv_bottom_3 = ConvBlock(
            in_channels=128,
            out_channels=256,
            n_convs=2,
            kernel_size=3,
            padding=0,
            downsample=True,
        )
        self.conv_bottom_4 = ConvBlock(
            in_channels=256,
            out_channels=out_channels,
            n_convs=2,
            kernel_size=3,
            padding=0,
            downsample=False,
        )

        self.recurrent = recurrent
        if self.recurrent:
            self.conv_rnn = ConvLSTMCell(out_channels, out_channels, 1)

        self.conv_lateral_3 = nn.Conv2d(
            in_channels=256, out_channels=out_channels, kernel_size=1, bias=True
        )
        self.conv_lateral_2 = nn.Conv2d(
            in_channels=128, out_channels=out_channels, kernel_size=1, bias=True
        )
        self.conv_lateral_1 = nn.Conv2d(
            in_channels=64, out_channels=out_channels, kernel_size=1, bias=True
        )
        self.conv_lateral_0 = nn.Conv2d(
            in_channels=32, out_channels=out_channels, kernel_size=1, bias=True
        )

        self.conv_dealias_3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv_dealias_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv_dealias_1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv_dealias_0 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv_out = nn.Sequential(
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                n_convs=1,
                kernel_size=3,
                padding=1,
                downsample=False,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
        )

        self.conv_bottleneck_out = nn.Sequential(
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                n_convs=1,
                kernel_size=3,
                padding=1,
                downsample=False,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
        )

    def reset(self):
        if self.recurrent:
            self.conv_rnn.reset()

    def forward(self, x):
        """
        :param x:
        :return: (highest res feature map, lowest res feature map)
        """

        # Bottom-up pathway
        c0 = self.conv_bottom_0(x)  # 31x31
        c1 = self.conv_bottom_1(c0)  # 23x23
        c2 = self.conv_bottom_2(c1)  # 15x15
        c3 = self.conv_bottom_3(c2)  # 5x5
        c4 = self.conv_bottom_4(c3)  # 1x1

        # Top-down pathway (with lateral cnx and de-aliasing)
        p4 = c4
        p3 = self.conv_dealias_3(
            self.conv_lateral_3(c3)
            + F.interpolate(p4, (c3.shape[2], c3.shape[3]), mode="bilinear")
        )
        p2 = self.conv_dealias_2(
            self.conv_lateral_2(c2)
            + F.interpolate(p3, (c2.shape[2], c2.shape[3]), mode="bilinear")
        )
        p1 = self.conv_dealias_1(
            self.conv_lateral_1(c1)
            + F.interpolate(p2, (c1.shape[2], c1.shape[3]), mode="bilinear")
        )
        p0 = self.conv_dealias_0(
            self.conv_lateral_0(c0)
            + F.interpolate(p1, (c0.shape[2], c0.shape[3]), mode="bilinear")
        )

        if self.recurrent:
            p0 = self.conv_rnn(p0)

        return self.conv_out(p0), self.conv_bottleneck_out(c4)


class JointEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(JointEncoder, self).__init__()

        self.conv1 = ConvBlock(
            in_channels=in_channels, out_channels=64, n_convs=2, downsample=True
        )
        self.conv2 = ConvBlock(
            in_channels=64, out_channels=128, n_convs=2, downsample=True
        )
        self.convlstm0 = ConvLSTMCell(128, 128, 3)
        self.conv3 = ConvBlock(
            in_channels=128, out_channels=256, n_convs=2, downsample=True
        )
        self.conv4 = ConvBlock(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=0,
            n_convs=1,
            downsample=False,
        )

        # Transformer Addition
        self.flatten = nn.Flatten()
        embed_dim = 256
        num_heads = 8
        self.multihead_attention0 = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

        self.prev_x_res = None
        self.gates = nn.Linear(2 * embed_dim, embed_dim)
        self.ls_layer = LayerScale(embed_dim)

        # Attention Mask Transformer
        self.fusion_layer0 = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1),
        )
        self.output_layers = nn.Sequential(nn.Linear(embed_dim, 512), nn.LeakyReLU(0.1))

    def reset(self):
        self.convlstm0.reset()
        self.prev_x_res = None

    def forward(self, x, attn_mask=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.convlstm0(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)

        if self.prev_x_res is None:
            self.prev_x_res = Variable(torch.zeros_like(x))

        x = self.fusion_layer0(torch.cat((x, self.prev_x_res), 1))

        x_attn = x[None, :, :].detach()
        if self.training:
            x_attn = self.multihead_attention0(
                query=x_attn, key=x_attn, value=x_attn, attn_mask=attn_mask.bool()
            )[0].squeeze(0)
        else:
            x_attn = self.multihead_attention0(query=x_attn, key=x_attn, value=x_attn)[
                0
            ].squeeze(0)
        x = x + self.ls_layer(x_attn)

        gate_weight = torch.sigmoid(self.gates(torch.cat((self.prev_x_res, x), 1)))
        x = self.prev_x_res * gate_weight + x * (1 - gate_weight)

        self.prev_x_res = x

        x = self.output_layers(x)

        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma

####测试
class JointEncoder_new(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(JointEncoder_new, self).__init__()

        self.conv1 = ConvBlock(
            in_channels=in_channels, out_channels=64, n_convs=2, downsample=True
        )
        self.conv2 = ConvBlock(
            in_channels=64, out_channels=128, n_convs=2, downsample=True
        )
        self.convlstm0 = ConvLSTM(128, 128, (3,3), 1)
        self.pooling = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3 = ConvBlock(
            in_channels=128, out_channels=256, n_convs=2, downsample=True
        )
        self.conv4 = ConvBlock(
            in_channels=256, out_channels=512, n_convs=2, downsample=True
        )
        self.convlstm1 = ConvLSTM(512, 512, (3,3), 1)
        self.conv5 = ConvBlock(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            padding=1,
            n_convs=2,
            downsample=False,
        )
        # self.conv4 = ConvBlock(
        #     in_channels=256,
        #     out_channels=256,
        #     kernel_size=3,
        #     padding=1,
        #     n_convs=1,
        #     downsample=False,
        # )

        # Transformer Addition
        # self.flatten0 = nn.Flatten(start_dim=2,end_dim=-1) #修改后，时间(深度)通道保留，xy像素通道展平，形状变成[4,256,x*x]
        self.flatten1 = nn.Flatten() #把所有通道展平
        # embed_dim = 784
        # num_heads = 8
        # self.multihead_attention0 = nn.MultiheadAttention(
        #     embed_dim, num_heads, batch_first=True
        # )

        # self.gates = nn.Linear(2 * embed_dim, embed_dim)

        # Attention Mask Transformer
        # self.fusion_layer0 = nn.Sequential(
        #     nn.Linear(embed_dim * 2, embed_dim),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.LeakyReLU(0.1),
        # )
        # self.output_layers1 = nn.Sequential(
        #     nn.Linear(embed_dim, 512), 
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(512,out_channels),
        #     nn.LeakyReLU(0.1)
        # )
        # self.output_layers2 = nn.Linear(256*out_channels,out_channels)
        self.output_layers3 = nn.Sequential(
            nn.Linear(512*9, 512), 
            nn.LeakyReLU(0.1),
            nn.Linear(512,out_channels),
            nn.LeakyReLU(0.1)
        )

    def reset(self):
        self.convlstm0.reset()
        self.prev_x_res = None


    def forward(self, x, attn_mask=None):
        x = self.conv1(x)
        x = self.conv2(x)
        # x,_ = self.convlstm0(x)
        x = self.pooling(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x,_ = self.convlstm1(x)
        x = self.pooling(x)
        x = self.conv5(x)
        # x = self.flatten0(x)      
    
        # x = self.multihead_attention0(query=x, key=x, value=x)[0].squeeze(0)

        # x = self.output_layers1(x)
        # x = self.flatten1(x)
        # x = self.output_layers2(x)
        x = self.flatten1(x)
        x = self.output_layers3(x)

        return x
