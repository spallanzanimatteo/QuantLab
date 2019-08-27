# Copyright (c) 2019 UniMoRe, Matteo Spallanzani
# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import importlib
import torch
import torch.nn as nn
import quantlab.nets as nets


def get_individual(logbook, _ckpt, _net):
    """Return a network for the experiment."""
    topology = importlib.import_module('quantlab.' + logbook.problem + '.topology')
    net      = getattr(topology, _net['class'])(**_net['params'])
    if _ckpt is not None:
        net.load_state_dict(_ckpt['net'])
    device   = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net      = net.to(device)
    if torch.cuda.device_count() > 1:
        net_maybe_par = nn.DataParallel(net)
    else:
        net_maybe_par = net

    # print network architecture
    if logbook.verbose:
        print('Network architecture: \t{}'.format(type(net).__name__))

    # print controllers
    if logbook.verbose:
        print('Controllers: \t{}'.format(str(nets.Controller.getControllers(net))))
        
    return net, net_maybe_par, device
