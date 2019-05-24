# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose

from quantlab.data.split import transform_random_split


_MNIST = {
    'Normalize': {
        'mean': (0.0,),
        'std':  (1.0,)
    }
}


def _get_transforms(augment):
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


def load_datasets(data_dir, augment, valid_fraction):
    transforms         = _get_transforms(augment)
    trainvalidset      = torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
    len_train          = int(len(trainvalidset) * (1.0 - valid_fraction))
    trainset, validset = transform_random_split(trainvalidset, [len_train, len(trainvalidset) - len_train],
                                                [transforms['training'], transforms['validation']])
    testset            = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms['validation'])
    return trainset, validset, testset
