from deform_part import deform_up, deform_down, deform_inconv, outconv, GCN, BR, SA_module
# from .roi_align.crop_and_resize import CropAndResizeFunction
# from roi_align.roi_align import RoIAlign
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import cv2
from torch.autograd import Variable
from scipy.ndimage.filters import gaussian_filter
import cv2
from PIL import Image
import os


def elastic_transform(image, alpha_range, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.

   # Arguments
       image: Numpy array with shape (height, width, channels).
       alpha_range: Float for fixed value or [lower, upper] for random value from uniform distribution.
           Controls intensity of deformation.
       sigma: Float, sigma of gaussian filter that smooths the displacement fields.
       random_state: `numpy.random.RandomState` object for generating displacement fields.
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    if np.isscalar(alpha_range):
        alpha = alpha_range
    else:
        alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

    n, c, h, w = image.size()
    shape = (h, w)
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    flow = np.array([dx, dy])
    output = np.zeros((n, 2, h, w))
    output[0, :, :, :] = flow
    output = torch.from_numpy(output)
    return output


def flow_warp(input, flow):
    # out_h, out_w = size
    n, c, h, w = input.size()
    out_h, out_w = h, w
    norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
    flow = flow.type_as(input).to(input.device)
    w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
    h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
    grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
    grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
    grid = grid + flow.permute(0, 2, 3, 1) / norm
    output = F.grid_sample(input, grid)
    return output


##################################################
##DUNet
##################################################
class DUNet_Global(nn.Module):
    def __init__(self, n_channels, n_classes, downsize_nb_filters_factor=1):
        super(DUNet_Global, self).__init__()
        self.inc = deform_inconv(n_channels, 64 // downsize_nb_filters_factor)
        self.down1 = deform_down(64 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor, p=0.2)
        self.down2 = deform_down(128 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor, p=0.2)
        self.down3 = deform_down(256 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor, p=0.2)
        self.down4 = deform_down(512 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor, p=0.2)
        self.up1 = deform_up(1024 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor, True, p=0.2)
        self.up2 = deform_up(512 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor, p=0.2)
        self.up3 = deform_up(256 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor, p=0.2)
        self.up4 = deform_up(128 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor, p=0.2)
        self.outc = outconv(64 // downsize_nb_filters_factor, n_classes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

    def forward(self, x):
        x1 = self.inc(x)
        # x1 = self.gcn1(x1)
        x_1 = self.relu1(x1)

        x2 = self.down1(x_1)
        # x2 = self.gcn2(x2)
        x_2 = self.relu2(x2)

        x3 = self.down2(x_2)
        # x3 = self.gcn3(x3)
        x_3 = self.relu3(x3)

        x4 = self.down3(x_3)
        x_4 = self.relu4(x4)

        x5 = self.down4(x_4)
        x_5 = self.relu5(x5)

        x6 = self.up1(x_5, x_4)
        # x_6 = self.relu6(x6)
        x7 = self.up2(x6, x_3)
        # x_7 = self.relu7(x7)
        x8 = self.up3(x7, x_2)
        # x_8 = self.relu8(x8)
        x9 = self.up4(x8, x_1)
        # x_9 = self.relu9(x9)
        x = self.outc(x9)
        return x, x1, x2, x3


class Network(nn.Module):
    def __init__(self, crop_margin=20, crop_prob=0.5, crop_sample_batch=1, n_channels=3, n_class=5, plane='X', TEST=None):
        super(RSTN, self).__init__()
        self.TEST = TEST
        self.margin = crop_margin
        self.prob = crop_prob
        self.batch = crop_sample_batch
        if plane == "Z":
            self.threshold = 1300
        elif plane == "Y":
            self.threshold = 90
        else:
            self.threshold = 80

        self.branch1 = DUNet_Global(n_channels, n_class)
        self.branch2 = DUNet_Global(n_channels, n_class)
        self.visuals = OrderedDict()

    def forward(self, image, label=None, mode=None, score=None, mask=None, lung_mask=None, epoch=0, step=1):

        _, _, He, Wi = image.size()
        h = image[:, :, : He, : Wi].contiguous()

        if self.TEST is None:
        
            if step == 1:
                f0_global, f1_global, f2_global, f3_global = self.branch1(h)
            if step == 2:
                f0_global, f1_global, f2_global, f3_global = self.branch2(h)
                
            h = f0_global
            coarse_prob = h
            return coarse_prob

        if self.TEST == 'C': 
        
            f0_global, _, _, _ = self.branch1(h)
            h = f0_global
            h = torch.softmax(h, dim=1)
            h = torch.argmax(h, dim=1)
            coarse_prob = h.squeeze()
            return coarse_prob




class IOU_loss(nn.Module):
    def __init__(self):
        super(IOU_loss, self).__init__()
        self.epsilon = 0.000001
        return

    def forward(self, pred, target):  # soft mode. per item.
        batch_num = pred.shape[0]
        pred = pred.contiguous().view(batch_num, -1)
        target = target.contiguous().view(batch_num, -1)
        DSC = (2 * (pred * target).sum(1) + self.epsilon) / \
              ((pred + target - pred * target).sum(1) + self.epsilon)
        return 1 - DSC.sum() / float(batch_num)


class NR_DSC_loss(nn.Module):
    def __init__(self):
        super(NR_DSC_loss, self).__init__()
        self.epsilon = 0.00001
        return

    def forward(self, pred, target):  # soft mode. per item.
        batch_num = pred.shape[0]
        pred = pred.contiguous().view(batch_num, -1)
        target = target.contiguous().view(batch_num, -1)
        sub = torch.abs(pred - target)
        sub = torch.pow(sub, 1.5)

        NR_DSC = (sub.sum(1)) / ((pred + target).sum(1) + self.epsilon)
        return NR_DSC.sum() / float(batch_num)


class DSC_loss(nn.Module):
    def __init__(self):
        super(DSC_loss, self).__init__()
        self.epsilon = 0.000001
        return

    def forward(self, pred, target):  # soft mode. per item.
        batch_num = pred.shape[0]
        pred = pred.contiguous().view(batch_num, -1)
        target = target.contiguous().view(batch_num, -1)
        DSC = (2 * (pred * target).sum(1) + self.epsilon) / \
              ((pred + target).sum(1) + self.epsilon)
        return 1 - DSC.sum() / float(batch_num)


class SA_DSC_loss(nn.Module):
    def __init__(self):
        super(SA_DSC_loss, self).__init__()
        self.epsilon = 0.000001
        return

    def forward(self, pred, target):  # soft mode. per item.
        batch_num = pred.shape[0]
        pred = pred.contiguous().view(batch_num, -1)
        target = target.contiguous().view(batch_num, -1)
        DSC = (2 * ((1 - pred) * pred * target).sum(1) + self.epsilon) / \
              (((1 - pred) * pred + target).sum(1) + self.epsilon)
        return 1 - DSC.sum() / float(batch_num)

class uncertainty_weighted_MSE_loss(nn.Module):
    def __init__(self):
        super(uncertainty_weighted_MSE_loss, self).__init__()
        return

    def forward(self, pred, target, uncertainty, num_class=5):
        pred = torch.softmax(pred, dim=1)
        weight1 = torch.exp(-uncertainty)
        weight1 = weight1.detach()
        MSE = (torch.mul(1 - weight1, (torch.abs(pred - target)) ** 2).sum(1))
        return MSE.sum() / (pred.shape[2] * pred.shape[3])

class MSE_loss(nn.Module):
    def __init__(self):
        super(MSE_loss, self).__init__()
        return

    def forward(self, pred, target, num_class=5):
        pred = torch.softmax(pred, dim=1)
        target = torch.softmax(target, dim=1)
        MSE = (torch.abs(pred - target) ** 2).sum(1)
        return MSE.sum() / (pred.shape[2] * pred.shape[3])

class uncertainty_weighted_CE_loss(nn.Module):
    def __init__(self):
        super(uncertainty_weighted_CE_loss, self).__init__()
        return

    def forward(self, pred, target, uncertainty, mean, num_class=5):
        # use uncertainty to generate alpha, small uncertainty means high quality prediction
        alpha = uncertainty * 0.8
        CE = 0
        pred = torch.softmax(pred, dim=1)
        pred = pred.squeeze(0)
        mean = mean.squeeze(0)
        target = target.squeeze(0)
        target = torch.nn.functional.one_hot(target.to(torch.int64), num_class)
        target = torch.transpose(torch.transpose(target, 0, 2), 1, 2)
        for i in range(0, num_class):
            soft_label = ((0.2 + alpha) * target[i, :, :]) + ((0.8-alpha) * mean[i, :, :])
            # gti = target[i,:,:]
            predi = pred[i, :, :]
            CE += -1.0 * soft_label * torch.log(torch.clamp(predi, 0.005, 1))
        CE = torch.mean(CE)
        return CE