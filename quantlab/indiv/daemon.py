# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import torch
import torch.nn as nn

from .transfer import load_pretrained


def get_topo(logbook):
    """Return a network for the experiment and the loss function for training."""
    
    # create the network
    net_config = logbook.config['indiv']['net']
    if net_config['class'] not in logbook.module.__dict__:
        raise ValueError('Network topology {} is not defined for problem {}'.format(net_config['class'], logbook.problem))
    net = getattr(logbook.module, net_config['class'])(**net_config['params'])
    
    # load checkpoint state or pretrained network
    if logbook.ckpt:
        net.load_state_dict(logbook.ckpt['indiv']['net'])
    elif net_config['pretrained']:
        load_pretrained(logbook, net)
        
    # move to proper device and, if possible, parallelize
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    net = net.to(device)
    if torch.cuda.device_count() > 1:
        net_maybe_par = nn.DataParallel(net)
    else:
        net_maybe_par = net
        
    # create the loss function
    loss_fn_config = logbook.config['indiv']['loss_function']
    loss_fn_dict = {**nn.__dict__, **logbook.module.__dict__}
    if loss_fn_config['class'] not in loss_fn_dict:
        raise ValueError('Loss function {} is not defined.'.format(loss_fn_config['class']))
    loss_fn = loss_fn_dict[loss_fn_config['class']]
    if 'net' in loss_fn.__init__.__code__.co_varnames:
        loss_fn = loss_fn(net, **loss_fn_config['params'])
    else:
        loss_fn = loss_fn(**loss_fn_config['params'])
        
    return net, net_maybe_par, device, loss_fn
