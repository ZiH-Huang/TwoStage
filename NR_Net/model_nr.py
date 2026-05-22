import torch
import torch.nn as nn
from deform_part_nr import deform_up, deform_down, deform_inconv, outconv
from collections import OrderedDict

##################################################
# Network
##################################################
class Net_Forward(nn.Module):
    def __init__(self, n_channels, n_classes, downsize_nb_filters_factor=1):
        super(Net_Forward, self).__init__()
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

        self.relu = nn.ReLU()

    def forward(self, x):
        # b, c, h, w = x.shape
        x1 = self.inc(x)
        x_1 = self.relu(x1)

        x2 = self.down1(x_1)
        x_2 = self.relu(x2)

        x3 = self.down2(x_2)
        x_3 = self.relu(x3)

        x4 = self.down3(x_3)
        x_4 = self.relu(x4)

        x5 = self.down4(x_4)
        x_5 = self.relu(x5)

        x6 = self.up1(x_5, x_4)
        x7 = self.up2(x6, x_3)
        x8 = self.up3(x7, x_2)
        x9 = self.up4(x8, x_1)
        x = self.outc(x9)
        return x


class Network(nn.Module):
    def __init__(self, n_channels=3, n_class=2, TEST=None):
        super(Network, self).__init__()
        self.TEST = TEST

        self.branch1 = Net_Forward(n_channels, n_class)
        self.branch2 = Net_Forward(n_channels, n_class)
        self.visuals = OrderedDict()

    def forward(self, image, step=1, if_test=0):

        _, _, He, Wi = image.size()
        h = image[:, :, : He, : Wi].contiguous()

        if if_test == 0:
            if step == 1:
                # main prediction
                x_predict = self.branch1(h)
            if step == 2:
                x_predict = self.branch2(h)

            model_prob = x_predict
            return model_prob

        if if_test == 1:
            x_predict = self.branch1(h)
            x_predict = torch.softmax(x_predict, dim=1)
            x_predict = torch.argmax(x_predict, dim=1)
            coarse_prob = x_predict.squeeze()
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

    def forward(self, pred, target, uncertainty, mean, num_class=2):
        # use uncertainty to generate alpha, small uncertainty means high quality prediction
        # alpha = uncertainty * 0.8
        alpha = uncertainty
        CE = 0
        pred = torch.softmax(pred, dim=1)
        pred = pred.squeeze(0)
        mean = mean.squeeze(0)
        target = target.squeeze(0)
        target = torch.nn.functional.one_hot(target.to(torch.int64), num_class)
        target = torch.transpose(torch.transpose(target, 0, 2), 1, 2)
        for i in range(0, num_class):
            # soft_label = ((0.2 + alpha) * target[i, :, :]) + ((0.8-alpha) * mean[i, :, :])
            soft_label = ((0.0 + alpha) * target[i, :, :]) + ((1.0 - alpha) * mean[i, :, :])
            # gti = target[i,:,:]
            predi = pred[i, :, :]
            CE += -1.0 * soft_label * torch.log(torch.clamp(predi, 0.00005, 1))
        CE = torch.mean(CE)
        return CE
