# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose

from quantlab.treat.data.split import transform_random_split


_MNIST = {
    'Normalize': {
        'mean': (0.0,),
        'std':  (1.0,)
    }
}


def get_transforms(augment):
    train_t = Compose([ToTensor(),
                       Normalize(**_MNIST['Normalize'])])
    valid_t = Compose([ToTensor(),
                       Normalize(**_MNIST['Normalize'])])
    if not augment:
        train_t = valid_t
    transforms = {
        'training':   train_t,
        'validation': valid_t
    }
    return transforms


def load_data_sets(dir_data, data_config):
    transforms           = get_transforms(data_config['augment'])
    trainvalid_set       = torchvision.datasets.MNIST(root=dir_data, train=True, download=True)
    len_train            = int(len(trainvalid_set) * (1.0 - data_config['valid_fraction']))
    train_set, valid_set = transform_random_split(trainvalid_set, [len_train, len(trainvalid_set) - len_train],
                                                [transforms['training'], transforms['validation']])
    test_set             = torchvision.datasets.MNIST(root=dir_data, train=False, download=True, transform=transforms['validation'])
    return train_set, valid_set, test_set
