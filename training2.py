# baseline1
# coding=utf-8
import os
import sys
import time
from utils import *
from model import *
import math
import torch
from PIL import Image
# from model_4stage_fusion import *
import torch.nn as nn
# from visualization import Visualizer
from collections import OrderedDict
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    data_path = sys.argv[1]
    current_fold = sys.argv[2]
    organ_number = int(sys.argv[3])
    low_range = int(sys.argv[4])
    high_range = int(sys.argv[5])
    slice_threshold = float(sys.argv[6])
    slice_thickness = int(sys.argv[7])
    organ_ID = int(sys.argv[8])
    plane = sys.argv[9]
    GPU_ID = int(sys.argv[10])
    learning_rate1 = float(sys.argv[11])
    learning_rate_m1 = int(sys.argv[12])
    learning_rate2 = float(sys.argv[13])
    learning_rate_m2 = int(sys.argv[14])
    crop_margin = int(sys.argv[15])
    crop_prob = float(sys.argv[16])
    crop_sample_batch = int(sys.argv[17])
    snapshot_path = os.path.join(snapshot_path, 'SIJ_training_' + \
                                 sys.argv[11] + 'x' + str(learning_rate_m1) + ',' + str(crop_margin))
    epoch = {}
    epoch['S'] = int(sys.argv[18])
    epoch['I'] = int(sys.argv[19])
    epoch['J'] = int(sys.argv[20])
    epoch['lr_decay'] = int(sys.argv[21])
    timestamp = sys.argv[22]

    def add_gauss(ori_image, mean_1=0.0, sigma_1=0.05, mean_2=1.0, sigma_2=0.01):
        _, ch, col, row = ori_image.shape
        gauss_1 = np.random.normal(mean_1, sigma_1, ori_image.numpy().shape)
        gauss_2 = np.random.normal(mean_2, sigma_2, ori_image.numpy().shape)
        # gauss = gauss.reshape(1, ch, col, row)
        noisy = np.multiply((ori_image + gauss_1), gauss_2)
        noisy = np.clip(noisy, 0, 1)
        return noisy

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    torch.cuda.manual_seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    pre_trained = False
    if pre_trained:
        snapshot_name = 'FD' + current_fold + ':' + \
                        plane + 'J' + str(slice_thickness) + '_' + str(organ_ID) + '_' + '20200423_211759'
        pre_weights = os.path.join(snapshot_path, snapshot_name) + '.pkl'
        if not os.path.isfile(pre_weights):
            raise RuntimeError(str(pre_weights) + 'not exists!!!')
        else:
            print('load ' + str(pre_weights))
    from Data import DataLayer

    training_set = DataLayer(data_path=data_path, current_fold=int(current_fold), organ_number=organ_number, \
                             low_range=low_range, high_range=high_range, slice_threshold=slice_threshold,
                             slice_thickness=slice_thickness, \
                             organ_ID=organ_ID, plane=plane)
    val_set = DataLayer(data_path=data_path, current_fold=int(current_fold), organ_number=organ_number, \
                        low_range=low_range, high_range=high_range, slice_threshold=slice_threshold,
                        slice_thickness=slice_thickness, \
                        organ_ID=organ_ID, plane=plane, testing=True)

    batch_size = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=True)
    print(current_fold + plane, len(trainloader))
    print(epoch)

    model = Network(crop_margin=crop_margin, crop_prob=crop_prob, crop_sample_batch=crop_sample_batch, plane=plane)
    snapshot = {}

    model_parameters = filter(lambda p: p.requires_grad, model.branch1.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('model parameters:', params)
    # two branches
    optimizer_l = torch.optim.Adam(filter(lambda p: p.requires_grad, model.branch1.parameters()), lr=learning_rate1, betas=(0.95, 0.999), weight_decay=0.00001)
    optimizer_r = torch.optim.Adam(filter(lambda p: p.requires_grad, model.branch2.parameters()), lr=learning_rate1, betas=(0.95, 0.999), weight_decay=0.00001)
    
    if pre_trained:
        model_dict = model.state_dict()
        pretrained_model = torch.load(pre_weights)
        pretrained_dict = {k: v
                           for k, v in pretrained_model.items()
                           if k in model_dict and k.startswith('coarse_model')}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(plane + 'load pre-trained Coarse model successfully!')

    criterion = DSC_loss()
    self_adjust_criterion = SA_DSC_loss()
    criterion_bce = nn.BCELoss()
    criterion_mse = MSE_loss()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss()
    criterion_wce = uncertainty_weighted_CE_loss()
    # MAE LOSS
    criterion_mae = nn.L1Loss()
    criterion_iou = IOU_loss()
    criterion_nr = NR_DSC_loss()
    COARSE_WEIGHT = 1 / 2.0

    model = model.cuda()
    # visualizer = Visualizer()
    errors_ret = OrderedDict()
    errors_val = OrderedDict()
    epoches = 0
    val_fine_loss = 100.0
    val_coarse_loss = 100.0
    for mode in ['S', 'I', 'J']:
        print('mode: ' + str(mode))
        try:
            for e in range(epoch[mode]):
                model.train()
                total_loss = 0.0
                total_coarse_loss = 0.0
                total_fine_loss = 0.0
                average_coarse_loss = 0.0
                average_fine_loss = 0.0
                total_coarse_bbox_loss = 0.0
                total_fine_bbox_loss = 0.0
                start = time.time()
                coarse_loss = 0.0
                for index, (image, label) in enumerate(tqdm(trainloader)):
                    start_it = time.time()
                    optimizer_l.zero_grad()
                    optimizer_r.zero_grad()
                    # label = label[:,0,:,:]
                    label = label.squeeze(dim=1)
                    #refine = refine.squeeze(dim=1)
                    image, label = image.float(), label.cuda().long()
                    # branch 2 image
                    # gauss_image = (add_gauss(image)).type(torch.FlaotTensor).cuda()
                    gauss_image = (add_gauss(image)).float().cuda()
                    # branch 1 image
                    image = image.float().cuda()
                    # epoch
                    epoch_ = math.floor(e / 2)
                    # coarse_prob 1*5*256*256
                    # prediction uncertainty 1*1*256*256
                    coarse_prob_l = model(image, label, mode=mode, epoch=epoch_, step=1)
                    coarse_prob_r = model(image, label, mode=mode, epoch=epoch_, step=2)
                    # variance uncertainty
                    h_var = torch.stack([F.softmax(coarse_prob_l, dim=1), F.softmax(coarse_prob_r, dim=1)], 0)
                    var_uncertainty = torch.sum(torch.std(h_var, dim=0), dim=1, keepdim=True)
                    var_uncertainty = (var_uncertainty - var_uncertainty.min())/(var_uncertainty.max() - var_uncertainty.min())
                    var_uncertainty = var_uncertainty.squeeze(0)
                    mean_prob = (F.softmax(coarse_prob_l, dim=1) + F.softmax(coarse_prob_r, dim=1))/2
                    # entropy uncertainty
                    # coarse_prob_max 1*256*256
                    loss_l_w = criterion_wce(coarse_prob_l, label, var_uncertainty, mean_prob)
                    loss_r_w = criterion_wce(coarse_prob_r, label, var_uncertainty, mean_prob)
                    loss_l = criterion_ce(coarse_prob_l, label)
                    loss_r = criterion_ce(coarse_prob_r, label)
                    loss_consistency = criterion_mse(coarse_prob_l, coarse_prob_r)
                    # coarse_loss_wce = criterion_wce(coarse_prob, refine, uncertainty)
                    # coarse_loss = criterion_ce(coarse_prob, refine)
                    if (e < 2):
                        loss = loss_l + loss_r
                    if (e >= 2):
                        # loss = loss_l_w + loss_r_w + 0.2 * loss_consistency
                        loss = loss_l_w+ loss_r_w
                    total_loss += loss.item()
                    total_coarse_loss += coarse_loss
                    loss.backward()
                    optimizer_l.step()
                    optimizer_r.step()

                    if (index + 1) % 20 == 0:
                        average_coarse_loss = total_coarse_loss

                    del image, label, coarse_prob_l, loss, loss_l, loss_r, loss_r_w, loss_l_w, coarse_prob_r, loss_consistency

                if e < 1:
                    print('lr decay1')
                    for param_group_r in optimizer_r.param_groups:
                        param_group_r['lr'] *= 2
                    for param_group_l in optimizer_l.param_groups:
                        param_group_l['lr'] *= 2

                if e >= 1:
                    print('lr decay2')
                    for param_group_r in optimizer_r.param_groups:
                        param_group_r['lr'] *= 0.95
                    for param_group_l in optimizer_l.param_groups:
                        param_group_l['lr'] *= 0.95

                print(current_fold + plane + mode, "Epoch[%d/%d], Total Coarse/Avg Loss %.4f/%.4f, Time elapsed %.2fs" \
                      % (e + 1, epoch[mode], total_coarse_loss / len(trainloader), total_loss / len(trainloader),
                         time.time() - start))

                ########################################validation#########################################################################
                model.eval()
                with torch.no_grad():
                    total_loss = 0.0
                    total_coarse_loss = 0.0
                    total_fine_loss = 0.0
                    average_coarse_loss = 0.0
                    average_fine_loss = 0.0
                    start = time.time()
                    for index, (image, label) in enumerate(tqdm(valloader)):
                        start_it = time.time()
                        label = label.squeeze(dim=1)
                        refine = refine.squeeze(dim=1)
                        image, label, refine = image.cuda().float(), label.cuda().long(), refine.cuda().long()
                        coarse_prob = model(image, refine, mode=mode, epoch=epoch_, step=1)
                        coarse_loss = criterion_ce(coarse_prob, refine)
                        loss = coarse_loss

                        total_loss += loss.item()
                        total_coarse_loss += coarse_loss.item()

                        if (index + 1) % 20 == 0:
                            errors_val['val_coarse_loss'] = (total_coarse_loss - average_coarse_loss) / 20
                            average_coarse_loss = total_coarse_loss
                        del image, label, coarse_prob, loss, coarse_loss

                    print(current_fold + plane + mode, "Epoch[%d/%d], val Coarse %.4f" \
                          % (e + 1, epoch[mode], total_coarse_loss / (len(valloader) + 1)))
                    if val_coarse_loss > total_coarse_loss / (len(valloader) + 1):
                        if mode == 'I':
                            val_coarse_loss = total_coarse_loss / (len(valloader) + 1)
                            snapshot_name = 'FD' + current_fold + ':' + \
                                            plane + 'J' + str(slice_thickness) + '_' + str(organ_ID) + '_' + timestamp
                            snapshot[mode] = os.path.join(snapshot_path, snapshot_name) + '.pkl'
                            torch.save(model.state_dict(), snapshot[mode])
                            snapshot[mode] = os.path.join(snapshot_path, snapshot_name) + '_entire.pkl'
                            torch.save(model, snapshot[mode])
                    if (epoches + 1) % 1 == 0:
                        snapshot_name = 'FD' + current_fold + ':' + \
                                        plane + mode + str(slice_thickness) + '_' + str(organ_ID) + '_' + timestamp
                        snapshot[mode] = os.path.join(snapshot_path, snapshot_name) + '_' + str(e + 1) + '.pkl'
                        torch.save(model.state_dict(), snapshot[mode])
                epoches += 1
        except KeyboardInterrupt:
            print('!' * 10, 'save before quitting ...')
        finally:
            snapshot_name = 'FD' + current_fold + ':' + \
                            plane + mode + str(slice_thickness) + '_' + str(organ_ID) + '_' + timestamp
            snapshot[mode] = os.path.join(snapshot_path, snapshot_name) + '_last.pkl'
            # if mode == 'S' or mode == 'J':
            print('#' * 10, 'end of ' + current_fold + plane + mode + ' training stage!')

