# Copyright (c) 2019 UniMoRe, Matteo Spallanzani
# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import torch
import torch.optim as optim
import torch.utils.data as tud

import itertools

from quantlab.treat.thermo.thermostat import Thermostat
import quantlab.treat.algo.lr_schedulers as lr_schedulers

class DynamicSubsetRandomSampler(tud.Sampler):
    r"""Samples a fixed number of elements randomly from a dataset of fixed size without replacement.

    Arguments:
        numSamples: the number of samples to take
        datasetLen: the size of the dataset from which to draw samples
    """

    def __init__(self, numSamples, datasetLen):
        assert(isinstance(datasetLen, int) or datasetLen.is_integer())
        assert(isinstance(numSamples, int) or numSamples.is_integer())
        self.datasetLen = datasetLen
        self.numSamples = numSamples

    def __iter__(self):
        numFullSets = self.numSamples // self.datasetLen
        numRemainingSamples = self.numSamples - numFullSets*self.datasetLen
        indexesAll = (i for i in list())
        for i in range(numFullSets):
            indexes = torch.randperm(self.datasetLen)
            indexesAll = itertools.chain(indexesAll, 
                                         (i.item() for i in indexes))
        indexes = torch.randperm(self.datasetLen)[:numRemainingSamples]
        indexesAll = itertools.chain(indexesAll, 
                                     (i.item() for i in indexes))
        return indexesAll

    def __len__(self):
        return self.numSamples

def get_algo(logbook, net):
    """Return a training procedure for the experiment."""
    # set ANA cooling schedule
    thr_config = logbook.config['treat']['thermostat']
    thr = Thermostat(net, **thr_config['params'])
    if logbook.ckpt:
        thr.load_state_dict(logbook.ckpt['treat']['thermostat'])
    # set algo algorithm
    opt_config      = logbook.config['treat']['optimizer']
    opt             = optim.__dict__[opt_config['class']](net.parameters(), **opt_config['params'])
    if logbook.ckpt:
        opt.load_state_dict(logbook.ckpt['treat']['optimizer'])
    lr_sched_config = logbook.config['treat']['lr_scheduler']
    lr_sched_dict   = {**optim.lr_scheduler.__dict__, **lr_schedulers.__dict__}
    lr_sched        = lr_sched_dict[lr_sched_config['class']](opt, **lr_sched_config['params'])
    if logbook.ckpt:
        lr_sched.load_state_dict(logbook.ckpt['treat']['lr_scheduler'])
    return thr, opt, lr_sched


def get_data(logbook, num_workers=10):
    """Return data for the experiment."""
    data_config = logbook.config['treat']['data']
    # make dataset random split consistent (to prevent training instances from filtering into validation set)
    rng_state = torch.get_rng_state()
    torch.manual_seed(1234)
    # load preprocessed datasets
    train_set, valid_set, test_set = logbook.module.load_data_sets(logbook.dir_data, data_config)
    # create random training set subselector for mini-epochs
    if 'epoch_size_train' in data_config.keys():
        shuffleTrain = False
        cfgVal = float(data_config['epoch_size_train'])
#        if cfgVal > 1:
#            assert(cfgVal.is_integer())
#            numSamples = int(cfgVal)
#        else:
        numSamples = int(cfgVal*len(train_set))
#        assert(numSamples <= len(train_set))
        samplerTrain = DynamicSubsetRandomSampler(numSamples, len(train_set))
    else:
        shuffleTrain, samplerTrain = True, None
    # create loaders
    if hasattr(train_set, 'collate_fn'):  # if one data set needs `collate`, all the data sets should
        train_l = tud.DataLoader(train_set, batch_size=data_config['bs_train'], 
                                 shuffle=shuffleTrain, sampler=samplerTrain, 
                                 num_workers=num_workers, 
                                 collate_fn=train_set.collate_fn)
        valid_l = tud.DataLoader(valid_set, batch_size=data_config['bs_valid'], 
                                 shuffle=True, num_workers=num_workers, 
                                 collate_fn=valid_set.collate_fn)
        test_l  = tud.DataLoader(test_set,  batch_size=data_config['bs_valid'], 
                                 shuffle=True, num_workers=num_workers, 
                                 collate_fn=test_set.collate_fn)
    else:
        train_l = tud.DataLoader(train_set, batch_size=data_config['bs_train'], 
                                 shuffle=shuffleTrain, sampler=samplerTrain, 
                                 num_workers=num_workers)
        valid_l = tud.DataLoader(valid_set, batch_size=data_config['bs_valid'], 
                                 shuffle=True, num_workers=num_workers)
        test_l  = tud.DataLoader(test_set, batch_size=data_config['bs_valid'], 
                                 shuffle=True, num_workers=num_workers)
    torch.set_rng_state(rng_state)
    return train_l, valid_l, test_l
