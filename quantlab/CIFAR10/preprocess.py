# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import torchvision
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, Compose

from quantlab.data.split import transform_random_split


_CIFAR10 = {
    'Normalize': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std':  (0.2470, 0.2430, 0.2610)
    }
}


def _get_transforms(augment):
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


def load_datasets(data_dir, augment, valid_fraction):
    transforms         = _get_transforms(augment)
    trainvalidset      = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    len_train          = int(len(trainvalidset) * (1.0 - valid_fraction))
    trainset, validset = transform_random_split(trainvalidset, [len_train, len(trainvalidset) - len_train],
                                                [transforms['training'], transforms['validation']])
    testset            = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transforms['validation'])
    return trainset, validset, testset
