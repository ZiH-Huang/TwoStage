# baseline1
# coding=utf-8
import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from Data import DataLayer
from collections import OrderedDict
from utils_nr import *
from model_nr import *

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
        noisy = np.multiply((ori_image + gauss_1), gauss_2)
        noisy = np.clip(noisy, 0, 1)
        return noisy


    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    torch.cuda.manual_seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True

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
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=16,
                                              drop_last=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=16,
                                            drop_last=True)
    print(current_fold + plane, len(trainloader))
    print(epoch)

    Network_model = Network()
    Network_snapshot = {}

    # two branches
    optimizer_l = torch.optim.Adam(filter(lambda p: p.requires_grad, Network_model.branch1.parameters()),
                                   lr=learning_rate1, betas=(0.95, 0.999), weight_decay=0.00001)
    optimizer_r = torch.optim.Adam(filter(lambda p: p.requires_grad, Network_model.branch2.parameters()),
                                   lr=learning_rate2, betas=(0.95, 0.999), weight_decay=0.00001)


    criterion_ce = nn.CrossEntropyLoss()
    criterion_wce = uncertainty_weighted_CE_loss()

    Network_model = Network_model.cuda()
    epoches = 0
    warm_up = 2

    val_fine_loss = 100.0
    val_coarse_loss = 100.0
    for mode in ['S', 'I', 'J']:
        try:
            for e in range(epoch[mode]):
                Network_model.train()
                total_loss = 0.0

                for index, (image, label) in enumerate(tqdm(trainloader)):
                    optimizer_l.zero_grad()
                    optimizer_r.zero_grad()

                    label = label.squeeze(dim=1)
                    image, label = image.float(), label.cuda().long()
                    # branch 2 image
                    gauss_image = (add_gauss(image)).float().cuda()
                    # branch 1 image
                    image = image.float().cuda()
                    # coarse_prob 1*5*256*256
                    # prediction uncertainty 1*1*256*256
                    coarse_prob_l = Network_model(image, step=1)
                    coarse_prob_r = Network_model(gauss_image, step=2)
                    # variance uncertainty

                    h_var = torch.stack([F.softmax(coarse_prob_l, dim=1), F.softmax(coarse_prob_r, dim=1)], 0)
                    var_uncertainty = torch.sum(torch.std(h_var, dim=0), dim=1, keepdim=True)
                    var_uncertainty = var_uncertainty.squeeze(0)
                    uncertainty = (var_uncertainty - var_uncertainty.min()) / (
                                var_uncertainty.max() - var_uncertainty.min() + 1e-8)

                    mean_prob = (F.softmax(coarse_prob_l, dim=1) + F.softmax(coarse_prob_r, dim=1)) / 2

                    # prediction loss
                    loss_l_w = criterion_wce(coarse_prob_l, label, uncertainty, mean_prob)
                    loss_r_w = criterion_wce(coarse_prob_r, label, uncertainty, mean_prob)
                    loss_l = criterion_ce(coarse_prob_l, label)
                    loss_r = criterion_ce(coarse_prob_r, label)

                    if (e < warm_up):
                        loss = loss_l + loss_r
                    else:
                        loss = loss_l_w + loss_r_w

                    total_loss += loss.item()
                    loss.backward()
                    optimizer_l.step()
                    optimizer_r.step()

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

                ########################################validation#########################################################################
                Network_model.eval()
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
                        image, label = image.cuda().float(), label.cuda().long()

                        coarse_prob = Network_model(image, step=1)
                        coarse_loss = criterion_ce(coarse_prob, label)
                        loss = coarse_loss

                        total_loss += loss.item()
                        total_coarse_loss += coarse_loss.item()


                    print(current_fold + plane + mode, "Epoch[%d/%d], val Coarse %.4f" \
                          % (e + 1, epoch[mode], total_coarse_loss / (len(valloader) + 1)))

                    if (epoches + 1) % 1 == 0:
                        snapshot_name = 'FD' + current_fold + ':' + \
                                        plane + mode + str(slice_thickness) + '_' + str(organ_ID) + '_' + timestamp
                        Network_snapshot[mode] = os.path.join(snapshot_path, snapshot_name) + '_' + str(e + 1) + '.pkl'
                        torch.save(Network_model.state_dict(), Network_snapshot[mode])
                epoches += 1

        except KeyboardInterrupt:
            print('!' * 10, 'save before quitting ...')

