# Copyright (c) 2019 UniMoRe, Matteo Spallanzani
# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import argparse

from quantlab.protocol.logbook import Logbook
from quantlab.indiv.daemon import get_topo
from quantlab.treat.daemon import get_algo, get_data
from quantlab.protocol.rooms import train, test
import quantlab.indiv as indiv


# Command Line Interface
parser = argparse.ArgumentParser(description='QuantLab')
parser.add_argument('--problem',    help='MNIST/CIFAR-10/ImageNet/COCO')
parser.add_argument('--topology',   help='Network topology')
parser.add_argument('--exp_id',     help='Experiment to launch/resume',           default=None)
parser.add_argument('--load',       help='Checkpoint to load: best/last/i_epoch', default='best')
parser.add_argument('--mode',       help='Experiment mode: train/test',           default='train')
parser.add_argument('--ckpt_every', help='Frequency of checkpoints (in epochs)',  default=10)
args = parser.parse_args()

# create/retrieve experiment logbook
logbook = Logbook(args.problem, args.topology, args.exp_id, args.load)

# create/retrieve network and treatment
net, net_maybe_par, device, loss_fn = get_topo(logbook)
thr, opt, lr_sched                  = get_algo(logbook, net)
train_l, valid_l, test_l            = get_data(logbook)

# run experiment
if args.mode == 'train':
    for _ in range(logbook.i_epoch + 1, logbook.config['treat']['max_epoch'] + 1):
        
        logbook.start_epoch()
        thr.step()
        
        #prepare training network
        net.train()
        for ctrlr in indiv.Controller.getControllers(net):
            # call controllers for e.g. LR, annealing, ... adjustments
            ctrlr.step_preTraining(logbook.i_epoch, opt, tensorboardWriter=logbook.writer)
            
        # validate pre-training network
        validPreTrain_stats = test(logbook, net, device, loss_fn, valid_l, valid=True, prefix='validPreTrain')
        
        # train
        train_stats = train(logbook, net_maybe_par, device, loss_fn, opt, train_l)
        
        # prepare validation network
        net.eval()
        for ctrlr in indiv.Controller.getControllers(net):
            ctrlr.step_preValidation(logbook.i_epoch, tensorboardWriter=logbook.writer)
            
        #validate (re-)trained network
        valid_stats = test(logbook, net, device, loss_fn, valid_l, valid=True)
        stats = {**train_stats, **valid_stats, **validPreTrain_stats}
        
        # update learning rate
        if 'metrics' in lr_sched.step.__code__.co_varnames:
            lr_sched_metric = stats[logbook.config['treat']['lr_scheduler']['step_metric']]
            lr_sched.step(lr_sched_metric)
        else:
            lr_sched.step()
            
        # save model if update metric has improved...
        if logbook.is_better(stats):
            ckpt = {'indiv': {'net': net.state_dict()},
                    'treat': {
                        'thermostat':   thr.state_dict(),
                        'optimizer':    opt.state_dict(),
                        'lr_scheduler': lr_sched.state_dict(),
                        'i_epoch':      logbook.i_epoch
                    },
                    'protocol': {'metrics': logbook.metrics}}
            logbook.store_checkpoint(ckpt, is_best=True)
            
        # ...and/or if checkpoint epoch
        is_ckpt_epoch = (logbook.i_epoch % int(args.ckpt_every)) == 0
        if is_ckpt_epoch:
            ckpt = {'indiv': {'net': net.state_dict()},
                    'treat': {
                        'thermostat':   thr.state_dict(),
                        'optimizer':    opt.state_dict(),
                        'lr_scheduler': lr_sched.state_dict(),
                        'i_epoch':      logbook.i_epoch
                    },
                    'protocol': {'metrics': logbook.metrics}}
            logbook.store_checkpoint(ckpt)
            
elif args.mode == 'test':
    # test
    net.eval()
    test_stats = test(logbook, net, device, loss_fn, test_l)
