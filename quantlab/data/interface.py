# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import importlib
import torch


def get_data(logbook, _data, num_workers=20):
    """Return data for the experiment."""
    # make dataset random split consistent (to prevent training instances from filtering into validation set)
    rng_state = torch.get_rng_state()
    torch.manual_seed(1234)
    # load preprocessed datasets
    preprocess = importlib.import_module('quantlab.' + logbook.problem + '.preprocess')
    trainset, validset, testset = preprocess.load_datasets(logbook.data_dir, **_data['preprocess'])
    # create loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=_data['batch_size_train'], shuffle=True,  num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(validset, batch_size=_data['batch_size_valid'], shuffle=False, num_workers=num_workers)
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=_data['batch_size_valid'], shuffle=False, num_workers=num_workers)
    torch.set_rng_state(rng_state)
    # print data properties
    if logbook.verbose:
        print('Data augmentation: \t{}'.format(_data['preprocess']['augment']))
    return trainloader, validloader, testloader
