# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import argparse

from quantlab.protocol.logbook import Logbook
from quantlab.nets.interface   import get_individual
from quantlab.treat.interface  import get_treatment
from quantlab.data.interface   import get_data
from quantlab.protocol.launch  import train, test


# Command Line Interface
parser = argparse.ArgumentParser(description='QuantLab')
parser.add_argument('--problem',    help='MNIST/CIFAR10/ImageNet')
parser.add_argument('--exp_id',     help='Experiment to launch/resume',           default=None)
parser.add_argument('--load',       help='Checkpoint to load: best/last/i_epoch', default='best')
parser.add_argument('--mode',       help='Experiment mode: train/test',           default='train')
parser.add_argument('--save_epoch', help='Frequency of checkpoints (in epochs)',  default=50)
args = parser.parse_args()

# create/retrieve experiment logbook
logbook = Logbook(args.problem, args.exp_id)

# create/retrieve individual, treatment and data
_net, _thr, _loss_fn, _opt, _lr_sched, _data = logbook.load_config()
_ckpt                                        = logbook.load_checkpoint(args.load)
net, net_maybe_par, device                   = get_individual(logbook, _ckpt, _net)
thr, loss_fn, opt, lr_sched                  = get_treatment(logbook, _ckpt, net, _thr, _loss_fn, _opt, _lr_sched)
trainloader, validloader, testloader         = get_data(logbook, _data)
meter                                        = logbook.load_status(_ckpt)

# run experiment
if args.mode == 'train':
    for _ in range(logbook.i_epoch, logbook.max_epoch + 1):
        thr.step()
        # train
        net.train()
        train_stats = train(logbook, meter, net_maybe_par, device, loss_fn, opt, trainloader)
        # validate
        net.eval()
        valid_stats = test(logbook, meter, net, device, loss_fn, validloader, valid=True)
        # update learning rate
        if 'metrics' in lr_sched.step.__code__.co_varnames:
            lr_sched_metric = {**train_stats, **valid_stats}[_lr_sched['update_stat']]
            lr_sched.step(lr_sched_metric)
        else:
            lr_sched.step()
        # save model if validation metric has improved and/or if checkpoint epoch
        if meter.is_better(valid_stats['valid_metric'], logbook.best_metric):
            logbook.best_metric = valid_stats['valid_metric']
            _ckpt = {'net':         net.state_dict(),
                     'thr':         thr.state_dict(),
                     'opt':         opt.state_dict(),
                     'lr_sched':    lr_sched.state_dict(),
                     'epoch':       logbook.i_epoch,
                     'best_metric': logbook.best_metric}
            logbook.store_checkpoint(_ckpt, is_best=True)
        if (logbook.i_epoch % args.save_epoch) == 0:
            _ckpt = {'net':         net.state_dict(),
                     'thr':         thr.state_dict(),
                     'opt':         opt.state_dict(),
                     'lr_sched':    lr_sched.state_dict(),
                     'epoch':       logbook.i_epoch,
                     'best_metric': logbook.best_metric}
            logbook.store_checkpoint(_ckpt)
        logbook.i_epoch += 1
elif args.mode == 'test':
    # test
    net.eval()
    test_stats = test(logbook, meter, net, device, loss_fn, testloader)
