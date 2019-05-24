# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import torch.nn as nn
import torch.optim as optim

from quantlab.treat.thermostat import Thermostat
import quantlab.treat.loss_function as loss_function
import quantlab.treat.lr_scheduler as lr_scheduler


def get_treatment(logbook, _ckpt, net, _thr, _loss_fn, _opt, _lr_sched):
    """Return a training procedure for the experiment."""
    loss_fn_dict  = {**nn.__dict__, **loss_function.__dict__}
    lr_sched_dict = {**optim.lr_scheduler.__dict__, **lr_scheduler.__dict__}
    if _ckpt is None:
        assert type(net).__name__ == _thr['class']
        thr        = Thermostat(net, **_thr['params'])
        loss_fn    = loss_fn_dict[_loss_fn['class']](**_loss_fn['params'])
        opt        = optim.__dict__[_opt['class']](net.parameters(), **_opt['params'])
        lr_sched   = lr_sched_dict[_lr_sched['class']](opt, **_lr_sched['params'])
    else:
        assert type(net).__name__ == _thr['class']
        thr        = Thermostat(net, **_thr['params'])
        thr.load_state_dict(_ckpt['thr'])
        loss_fn    = loss_fn_dict[_loss_fn['class']](**_loss_fn['params'])
        opt        = optim.__dict__[_opt['class']](net.parameters(), **_opt['params'])
        opt.load_state_dict(_ckpt['opt'])
        lr_sched   = lr_sched_dict[_lr_sched['class']](opt, **_lr_sched['params'])
        lr_sched.load_state_dict(_ckpt['lr_sched'])
    # print training procedure
    if logbook.verbose:
        print('Loss function: \t\t{}'.format(type(loss_fn).__name__))
        print('Optimizer:     \t\t{}'.format(type(opt).__name__))
        print('LR scheduler:  \t\t{}'.format(type(lr_sched).__name__))
    return thr, loss_fn, opt, lr_sched
