# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import torchvision
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, Compose

from quantlab.treat.data.split import transform_random_split


_CIFAR10 = {
    'Normalize': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std':  (0.2470, 0.2430, 0.2610)
    }
}


def get_transforms(augment):
    train_t = Compose([RandomCrop(32, padding=4),
                       RandomHorizontalFlip(),
                       ToTensor(),
                       Normalize(**_CIFAR10['Normalize'])])
    valid_t = Compose([ToTensor(),
                       Normalize(**_CIFAR10['Normalize'])])
    if not augment:
        train_t = valid_t
    transforms = {
        'training':   train_t,
        'validation': valid_t
    }
    return transforms


def load_data_sets(dir_data, data_config):
    transforms           = get_transforms(data_config['augment'])
    trainvalid_set       = torchvision.datasets.CIFAR10(root=dir_data, train=True, download=True)
    if 'useTestForVal' in data_config.keys() and data_config['useTestForVal'] == True:
        train_set, valid_set = transform_random_split(trainvalid_set, 
                                                      [len(trainvalid_set), 0],
                                            [transforms['training'], transforms['validation']])
        test_set = torchvision.datasets.CIFAR10(root=dir_data, train=False, 
                                                download=True, 
                                                transform=transforms['validation'])
        valid_set = test_set
        print('using test set for validation.')
    else:
        len_train = int(len(trainvalid_set) * (1.0 - data_config['valid_fraction']))
        train_set, valid_set = transform_random_split(trainvalid_set, 
                                                      [len_train, len(trainvalid_set) - len_train],
                                                      [transforms['training'], transforms['validation']])
        test_set = torchvision.datasets.CIFAR10(root=dir_data, train=False, 
                                                download=True, 
                                                transform=transforms['validation'])
    return train_set, valid_set, test_set
