import numpy as np
import os
import csv
import ast
import sys
import nibabel


def is_organ(label, organ_ID):
    return label == organ_ID


def id_splits_iterator(splits_file, code=None):
    with open(splits_file, 'r') as f:
        r = csv.reader(f, delimiter=',', quotechar='"')
        headers = next(r)
        for row in r:
            if len(row) == 0:
                break
            if row[0].startswith('#'):
                continue
            spl = {headers[i].strip(): ast.literal_eval(row[i]) for i in range(len(headers))}

        if code is not None:
            return spl[code]
        else:
            return spl


def in_training_set(filename, splits_file):
    training_set = id_splits_iterator(splits_file, code='training')
    return filename in training_set


def in_testing_set(filename, splits_file):
    testing_set = id_splits_iterator(splits_file, code='testing')
    return filename in testing_set


def training_set_filename(current_fold=0):
    return os.path.join(list_path, 'training.txt')


def testing_set_filename(current_fold=0):
    return os.path.join(list_path, 'testing.txt')


def log_filename(snapshot_directory):
    count = 0
    while True:
        count += 1
        if count == 1:
            log_file_ = os.path.join(snapshot_directory, 'log.txt')
        else:
            log_file_ = os.path.join(snapshot_directory, 'log' + str(count) + '.txt')
        if not os.path.isfile(log_file_):
            return log_file_


def snapshot_name_from_timestamp(snapshot_path, \
                                 current_fold, plane, stage_code, slice_thickness, organ_ID, timestamp):
    snapshot_prefix = 'FD' + str(current_fold) + ':' + plane + \
                      stage_code + str(slice_thickness) + '_' + str(organ_ID)
    if len(timestamp) == 15:
        snapshot_prefix = snapshot_prefix + '_' + timestamp
    snapshot_name = snapshot_prefix + '.pkl'
    if os.path.isfile(os.path.join(snapshot_path, snapshot_name)):
        return snapshot_name
    else:
        return ''


def result_name_from_timestamp(result_path, current_fold, \
                               plane, stage_code, slice_thickness, organ_ID, volume_list, timestamp):
    result_prefix = 'FD' + str(current_fold) + ':' + plane + \
                    stage_code + str(slice_thickness) + '_' + str(organ_ID)
    if len(timestamp) == 15:
        result_prefix = result_prefix + '_' + timestamp
    result_name = result_prefix + '.pkl'
    if os.path.exists(os.path.join(result_path, result_name, 'volumes')):
        return result_name
    else:
        return ''


def volume_filename_testing(result_directory, t, i):
    return os.path.join(result_directory, str(t) + '_' + str(i + 1) + '.npz')


def volume_filename_fusion(result_directory, code, i):
    return os.path.join(result_directory, code + '_' + str(i + 1) + '.npz')


def volume_filename_coarse2fine(result_directory, r, i):
    return os.path.join(result_directory, 'R' + str(r) + '_' + str(i + 1) + '.npz')


def volume_filename_coarse2fine_thr(result_directory, r, i):
    return os.path.join(result_directory, 'Thr' + str(r) + '_' + str(i + 1) + '.npz')


def post_processing(F, S, threshold, organ_ID, lung_name, perfect_lung=True, post=True):
    if post:
        lung = nibabel.load(lung_name).get_data().transpose(1, 0, 2)
        if perfect_lung:
            F[lung == 0] = 0
        else:
            a = np.where(lung == 1)
            x0, x1 = np.min(a[0]), np.max(a[0])
            y0, y1 = np.min(a[1]), np.max(a[1])
            z0, z1 = np.min(a[2]), np.max(a[2])
            lung[x0:x1, y0:y1, z0:z1] = 1
            F[lung == 0] = 0
    return F


data_path = sys.argv[1]
ori_data_path = "/data/covid-19-data/"

image_path = os.path.join(ori_data_path, 'image')
image_path_ = {}
for plane in ['X', 'Y', 'Z']:
    image_path_[plane] = os.path.join(data_path, 'images_' + plane)
    if not os.path.exists(image_path_[plane]):
        os.makedirs(image_path_[plane])

label_path = os.path.join(ori_data_path, 'label')
label_path_ = {}
for plane in ['X', 'Y', 'Z']:
    label_path_[plane] = os.path.join(data_path, 'labels_' + plane)
    if not os.path.exists(label_path_[plane]):
        os.makedirs(label_path_[plane])

list_path = os.path.join(data_path, 'lists_HUnet')
if not os.path.exists(list_path):
    os.makedirs(list_path)
list_training = {}
for plane in ['X', 'Y', 'Z']:
    list_training[plane] = os.path.join(list_path, 'training_' + plane + '.txt')
model_path = os.path.join(data_path, 'models_HUnet')
if not os.path.exists(model_path):
    os.makedirs(model_path)
pretrained_model_path = os.path.join(data_path, 'models_HUnet', 'pretrained')
if not os.path.exists(pretrained_model_path):
    os.makedirs(pretrained_model_path)
snapshot_path = os.path.join(data_path, 'models_HUnet', 'snapshots')
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)
log_path = os.path.join(data_path, 'logs_HUnet')
if not os.path.exists(log_path):
    os.makedirs(log_path)
result_path = os.path.join(data_path, 'results_HUnet')
if not os.path.exists(result_path):
    os.makedirs(result_path)
