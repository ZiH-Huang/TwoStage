import torch.nn as nn
import torch.nn.functional as F
import torch

'''
from dcn import (DeformConv, DeformRoIPooling, DeformRoIPoolingPack,
                  ModulatedDeformRoIPoolingPack, ModulatedDeformConv,
                  ModulatedDeformConvPack, deform_conv, modulated_deform_conv,
                  deform_roi_pooling)
'''

'''from dcn import (DeformConv, DeformRoIPooling, 
                  ModulatedDeformRoIPoolingPack, ModulatedDeformConv,
                  ModulatedDeformConvPack) '''


class double_deform_conv(nn.Module):
    '''(conv => BN => ReLU) * '''

    def __init__(self, in_ch, out_ch,p=0):
        super(double_deform_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.Dropout(p),
            nn.GroupNorm(16, out_ch, eps=1e-05, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.Dropout(p),
            # ModulatedDeformConvPack(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, out_ch, eps=1e-05, affine=True),
            )
    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,p=0):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.Dropout(p),
            nn.GroupNorm(16, out_ch, eps=1e-05, affine=True),
            nn.ReLU(inplace=True)
            # nn.Conv2d(out_ch, out_ch, 3, padding=1),
            # nn.GroupNorm(out_ch//32, out_ch, eps=1e-05, affine=True),
            # nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class deform_inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(deform_inconv, self).__init__()
        self.conv = double_deform_conv(in_ch,out_ch)
    def forward(self, x):
        x = self.conv(x)
        return x


class deform_down(nn.Module):
    def __init__(self, in_ch, out_ch,p=0):
        super(deform_down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_deform_conv(in_ch, out_ch,p=p)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

'''
class deform_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(deform_up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = double_conv(int(1.5*in_ch), out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch , in_ch // 2, 2, stride=2)
            self.conv = double_conv(out_ch, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 1)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.conv2(x2)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        assert x1.shape == x2.shape
        x = x1+x2 #torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

'''

class deform_up(nn.Module):
    def __init__(self, in_ch, out_ch,innermost=False,bilinear=False,p=0):
        super(deform_up, self).__init__()
        if bilinear:
            if innermost:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                self.conv = double_conv(in_ch, out_ch,p=p)
            else:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                self.conv = double_conv(int(1.5*in_ch), out_ch,p=p)
        else:
            if innermost:
                self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            else:
                self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            self.conv = double_conv(out_ch, out_ch,p=p)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.conv2(x2)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        assert x1.shape == x2.shape
        x = x1+x2 #torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv1 =nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        return x1


class GCN(nn.Module):
    def __init__(self, in_channels, k=7):
        super(GCN, self).__init__()

        pad = (k-1) // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=(1, k), padding=(0, pad)),
            nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels, in_channels, kernel_size=(k, 1), padding=(pad, 0)),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=(k, 1), padding=(pad, 0)),
            nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels, in_channels, kernel_size=(1, k), padding=(0, pad)),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, in_channels,1))                      
    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)

        assert x1.shape == x2.shape
        x3 = self.conv3(x1+x2)
        return x+x3


class SA_module(nn.Module):
    #spatial attention module ---cross modal supervision
    def __init__(self, in_ch=64, num_group=1):
        super(SA_module, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,in_ch//16,kernel_size=1, stride=1, padding=0,dilation=1),
            nn.Conv2d(in_ch//16, in_ch//16, kernel_size=3, stride=1, padding=4,dilation=4),
            nn.Conv2d(in_ch//16, in_ch//16, kernel_size=3, stride=1, padding=4,dilation=4),
            nn.Conv2d(in_ch//16, 1,kernel_size=1, stride=1, padding=0,dilation=1))

    def forward(self, x_F, x_C):
        w1 = torch.sigmoid(self.conv(x_F))
        w2 = torch.sigmoid(self.conv(x_C))
        #residual fusion
        # x_V = x_V + x_V.mul(w)
        # x_A = x_A + x_A.mul(w)
        x_F = x_F.mul(w2)
        x_C = x_C.mul(w1)
        return x_F+x_C


class BR(nn.Module):
    def __init__(self, out_ch):
        super(BR, self).__init__()
        self.conv = nn.Conv2d(out_ch, out_ch,1) 
    def forward(self, x):
        x = self.conv(x)
        return x
        
class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)