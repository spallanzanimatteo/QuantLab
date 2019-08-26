# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import os
import numpy as np
import torch
import urllib.request

from yolov3tinybaseline import YOLOv3TinyBaseline


def yolov3tiny(verbose=False, new_file='yolov3-tiny.ckpt'):
    pretrained_file = 'https://pjreddie.com/media/files/yolov3-tiny.weights'
    new_file = os.path.join(os.pardir, os.pardir, os.pardir, 'COCO', 'YOLOv3Tiny', 'pretrained', new_file)
    net = YOLOv3TinyBaseline()
    parameters_list = [
        (4, 'phi01_conv.weight'),
        (1, 'phi01_bn.weight'),
        (0, 'phi01_bn.bias'),
        (2, 'phi01_bn.running_mean'),
        (3, 'phi01_bn.running_var'),
        (9, 'phi02_conv.weight'),
        (6, 'phi02_bn.weight'),
        (5, 'phi02_bn.bias'),
        (7, 'phi02_bn.running_mean'),
        (8, 'phi02_bn.running_var'),
        (14, 'phi03_conv.weight'),
        (11, 'phi03_bn.weight'),
        (10, 'phi03_bn.bias'),
        (12, 'phi03_bn.running_mean'),
        (13, 'phi03_bn.running_var'),
        (19, 'phi04_conv.weight'),
        (16, 'phi04_bn.weight'),
        (15, 'phi04_bn.bias'),
        (17, 'phi04_bn.running_mean'),
        (18, 'phi04_bn.running_var'),
        (24, 'phi05_conv.weight'),
        (21, 'phi05_bn.weight'),
        (20, 'phi05_bn.bias'),
        (22, 'phi05_bn.running_mean'),
        (23, 'phi05_bn.running_var'),
        (29, 'phi06_conv.weight'),
        (26, 'phi06_bn.weight'),
        (25, 'phi06_bn.bias'),
        (27, 'phi06_bn.running_mean'),
        (28, 'phi06_bn.running_var'),
        (34, 'phi07_conv.weight'),
        (31, 'phi07_bn.weight'),
        (30, 'phi07_bn.bias'),
        (32, 'phi07_bn.running_mean'),
        (33, 'phi07_bn.running_var'),
        (39, 'phi08_conv.weight'),
        (36, 'phi08_bn.weight'),
        (35, 'phi08_bn.bias'),
        (37, 'phi08_bn.running_mean'),
        (38, 'phi08_bn.running_var'),
        (44, 'phi09a_conv.weight'),
        (41, 'phi09a_bn.weight'),
        (40, 'phi09a_bn.bias'),
        (42, 'phi09a_bn.running_mean'),
        (43, 'phi09a_bn.running_var'),
        (46, 'phi10a_conv.weight'),
        (45, 'phi10a_conv.bias'),
        (51, 'phi09b_conv.weight'),
        (48, 'phi09b_bn.weight'),
        (47, 'phi09b_bn.bias'),
        (49, 'phi09b_bn.running_mean'),
        (50, 'phi09b_bn.running_var'),
        (56, 'phi10b_conv.weight'),
        (53, 'phi10b_bn.weight'),
        (52, 'phi10b_bn.bias'),
        (54, 'phi10b_bn.running_mean'),
        (55, 'phi10b_bn.running_var'),
        (58, 'phi11b_conv.weight'),
        (57, 'phi11b_conv.bias')
    ]
    response = urllib.request.urlopen(pretrained_file)
    data = response.read()
    parameters = np.frombuffer(data, dtype=np.float32, offset=5*4)  # first 5 * np.int32 are header
    new_dict = {}
    ptr = 0
    for (_, p) in sorted(parameters_list, key=lambda t: t[0]):
        n = net.state_dict()[p].numel()
        new_dict[p] = torch.from_numpy(parameters[ptr:ptr+n]).view_as(net.state_dict()[p])
        ptr += n
    assert ptr == len(parameters)
    torch.save({'indiv': {'net': new_dict}}, new_file)

    if verbose:
        for (_, p) in parameters_list:
            p_net = net.state_dict()[p]
            print('{:>30} \t {:>10} \t {}'.format(p, p_net.numel(), list(p_net.size())))

        for (_, p) in parameters_list:
            p_new = new_dict[p]
            print('{:>30} \t {:>10} \t {}'.format(p, p_new.numel(), list(p_new.size())))

        for (_, p) in parameters_list:
            p_net = net.state_dict()[p]
            p_new = new_dict[p]
            print('{:>30} \t {:>10}'.format(p, p_net.numel() - p_new.numel()))


if __name__ == '__main__':
    yolov3tiny()
