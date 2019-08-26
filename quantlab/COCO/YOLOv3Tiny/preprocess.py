# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import cv2
import numpy as np
import os
import random
from skimage import io  # conda install -c conda-forge scikit-image
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from ..utils.utils import xyxy2xywh


_img_formats_ = ['.bmp', '.jpg', '.jpeg', '.png', '.tif']


class COCODataset(Dataset):

    def __init__(self, index_file, augment=False, detect_corrupted_images=False):
        self.augment = augment
        self.img_size = 416  # must be in {i * 32, i = 10, 11, ..., 19}
        with open(index_file, 'r') as file:
            self.img_files = [x for x in file.read().splitlines() if os.path.splitext(x)[-1].lower() in _img_formats_]
        self.label_files = [x.replace('images', 'labels') for x in self.img_files]
        for img_format in _img_formats_:
            self.label_files = [x.replace(img_format, '.txt') for x in self.label_files]
        if detect_corrupted_images:
            for file in tqdm(self.img_files, desc='Detecting corrupted images'):
                try:
                    _ = io.imread(file)
                except:
                    print('Corrupted image detected: %s' % file)

    def __len__(self):
        return len(self.img_files)

    def _to_square(self, img, new_shape=416, color=(128, 128, 128)):
        # compute dilation factor (ratio) necessary to fit the image into the given square "container"
        # ASPECT RATIO SHOULD BE PRESERVED!
        shape = img.shape[:2]  # current shape [height, width]
        ratio = float(new_shape) / max(shape)
        rw, rh = ratio, ratio
        new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
        # compute padding
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)  # INTER_AREA is better, INTER_LINEAR is faster
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, rw, rh, dw, dh

    def __getitem__(self, index):
        # load image
        img_path = self.img_files[index]
        img = cv2.imread(img_path)  # BGR
        assert img is not None, 'Image not found: {}'.format(img_path)
        h, w, _ = img.shape
        img, rw, rh, dw, dh = self._to_square(img, new_shape=self.img_size)
        h_padded, w_padded, _ = img.shape
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR-to-RGB
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = torch.from_numpy(img)
        img /= 255.0
        # load label
        label = []
        gt_label_path = self.label_files[index]
        if os.path.exists(gt_label_path):
            x = torch.from_numpy(np.loadtxt(gt_label_path).reshape(-1, 5))
            if len(x) > 0:
                # annotations are in format (cls_id, x_rel, y_rel, w_rel, h_rel)
                label = torch.zeros_like(x)
                label[:, 0] = x[:, 0]
                # rescale bbox annotations wrt rescaled image
                label[:, 1] = rw * w * (x[:, 1] - x[:, 3] / 2) + dw
                label[:, 2] = rh * h * (x[:, 2] - x[:, 4] / 2) + dh
                label[:, 3] = rw * w * (x[:, 1] + x[:, 3] / 2) + dw
                label[:, 4] = rh * h * (x[:, 2] + x[:, 4] / 2) + dh
                label[:, 1:5] = xyxy2xywh(label[:, 1:5])
                # bbox annotations are in format (x_abs, y_abs, w_abs, h_abs)
                label[:, [1, 3]] /= w_padded
                label[:, [2, 4]] /= h_padded
                # bbox annotations are in format (x_rel, y_rel, w_rel, h_rel)
        gt_label = torch.zeros((len(label), 7))
        if len(label):
            gt_label[:, 2:] = label  # batch size, image id, annotation
        return img, gt_label

    def collate_fn(self, batch):
        inputs, gt_labels = list(zip(*batch))
        inputs = torch.stack(inputs, 0)
        bs = len(inputs)
        gt_labels = [gt for gt in gt_labels]
        for i, gt in enumerate(gt_labels):
            # CAVEAT: if every label is empty (zero annotations), batch information is lost and there will be a problem
            # transforming tensor label into list of tensor labels in `postprocess_gt` function (postprocess.py)
            if len(gt):
                # mark every annotation in the label with batch size and image id
                gt[:, 0] = bs
                gt[:, 1] = i
        gt_labels = torch.cat(gt_labels, 0)
        return inputs, gt_labels


def load_data_sets(dir_data, data_config):
    train_set = COCODataset(os.path.join(dir_data, 'trainvalno5k.txt'), augment=data_config['augment'])
    valid_set = COCODataset(os.path.join(dir_data, '5k.txt'))
    test_set  = valid_set
    return train_set, valid_set, test_set
