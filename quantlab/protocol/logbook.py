# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import importlib
import json
import os
import shutil
import sys
from tensorboardX import SummaryWriter
import torch


class Logbook(object):

    def __init__(self, problem, exp_id, verbose=True):
        self.verbose     = verbose
        self.problem     = problem
        self.prob_dir, self.data_dir, self.exp_dir, self.save_dir, self.stats_dir = self._get_folders(problem, exp_id)
        self.writer      = SummaryWriter(log_dir=self.stats_dir)
        self.config_file = self._get_config_file()
        self.max_epoch   = None
        self.i_epoch     = None
        self.best_metric = None

    def _get_folders(self, problem, exp_id):
        """Return a logbook for the experiment."""
        with open('hard_folders.json', 'r') as fp:
            d = json.load(fp)
            _HARD_DATA_STORAGE = d['data']
            _HARD_LOG_STORAGE  = d['log']
        # get pointers to SHARED resources
        src_dir  = sys.path[0]
        prob_dir = os.path.join(src_dir, problem)
        data_dir = os.path.join(prob_dir, 'data')
        log_dir  = os.path.join(prob_dir, 'log')
        # get pointers to SHARED storage (for data)
        hard_prob_dir_data = os.path.join(_HARD_DATA_STORAGE, problem)
        if not os.path.isdir(hard_prob_dir_data):
            os.mkdir(hard_prob_dir_data)
        hard_data_dir = os.path.join(hard_prob_dir_data, 'data')
        if not os.path.isdir(hard_data_dir):
            os.mkdir(hard_data_dir)
        if not os.path.isdir(data_dir):
            os.symlink(hard_data_dir, data_dir)
        # get pointers to SHARED storage (for log)
        hard_prob_dir_log = os.path.join(_HARD_LOG_STORAGE, problem)
        if not os.path.isdir(hard_prob_dir_log):
            os.mkdir(hard_prob_dir_log)
        hard_log_dir = os.path.join(hard_prob_dir_log, 'log')
        if not os.path.isdir(hard_log_dir):
            os.mkdir(hard_log_dir)
        if not os.path.isdir(log_dir):
            os.symlink(hard_log_dir, log_dir)
        # get pointers to PRIVATE experiment sections
        if exp_id is None:
            # create a new report
            exp_folders = [f for f in os.listdir(log_dir) if f.startswith('exp')]
            if len(exp_folders) == 0:
                exp_id = 0
            else:
                exp_id = max(int(f.replace('exp', '')) for f in exp_folders) + 1
        else:
            # retrieve an existing report
            exp_id = int(exp_id)
        exp_dir   = os.path.join(log_dir, 'exp'+str(exp_id).rjust(2, '0'))
        save_dir  = os.path.join(exp_dir, 'save')
        stats_dir = os.path.join(exp_dir, 'stats')
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)
        if not os.path.isdir(stats_dir):
            os.mkdir(stats_dir)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        if self.verbose:
            # print setup message
            message  = 'EXPERIMENT LOGBOOK\n'
            message += 'Problem directory:     \t{}\n'.format(prob_dir)
            message += 'Data directory:        \t{}\n'.format(data_dir)
            message += 'Experiment directory:  \t{}\n'.format(exp_dir)
            message += 'Checkpoints directory: \t{}\n'.format(save_dir)
            message += 'Statistics directory:  \t{}\n'.format(stats_dir)

            def print_message(message):
                """Print a nice delimiter around a multiline message."""
                lines = message.splitlines()
                tab_size = 4
                width = max(len(l) for l in lines) + tab_size
                print('+' + '-' * width + '+')
                for l in lines:
                    print(l)
                print('+' + '-' * width + '+')

            print_message(message)
        return prob_dir, data_dir, exp_dir, save_dir, stats_dir

    def _get_config_file(self):
        config_file = os.path.join(self.exp_dir, 'config.json')
        if not os.path.isfile(config_file):
            shutil.copyfile(os.path.join(self.prob_dir, 'config.json'), config_file)
            # write configuration to TensorBoard event file
            with open(config_file) as fp:
                d = json.load(fp)
            for k, v in d.items():
                self.writer.add_text(str(k), str(v))
        return config_file

    def load_config(self):
        """Load a configuration for the experiment (network, training procedure, data)."""
        with open(self.config_file, 'r') as fp:
            d = json.load(fp)
        _net           = d['architecture']
        _thr           = d['thermostat']
        _loss_fn       = d['loss_fn']
        _opt           = d['optimizer']
        _lr_sched      = d['lr_scheduler']
        self.max_epoch = d['max_epoch']
        _data          = d['data']
        return _net, _thr, _loss_fn, _opt, _lr_sched, _data

    def load_checkpoint(self, load):
        """Load states of network and training procedure."""
        # get checkpoint file name
        ckpt_list = os.listdir(self.save_dir)
        if len(ckpt_list) != 0:
            if load == 'best':
                ckpt_file = os.path.join(self.save_dir, 'best.ckpt')
            elif load == 'last':
                ckpt_file = max([os.path.join(self.save_dir, c) for c in ckpt_list], key=os.path.getctime)
            else:
                ckpt_list.remove('best.ckpt')
                ckpt_epochs = set(int(c.split('.')[0].replace('epoch', '')) for c in ckpt_list)
                if int(load) in ckpt_epochs:
                    ckpt_name = 'epoch'+load.rjust(4, '0')+'.ckpt'
                    ckpt_file = os.path.join(self.save_dir, ckpt_name)
                else:
                    ckpt_file = None
        else:
            ckpt_file = None
        # get checkpoint
        if ckpt_file:
            _ckpt = torch.load(ckpt_file)
        else:
            _ckpt = None
        # print checkpoint message
        if self.verbose:
            if ckpt_file:
                print('Checkpoint found: \t{}'.format(ckpt_file))
            else:
                print('No checkpoint at: \t{}'.format(self.save_dir))
        return _ckpt

    def store_checkpoint(self, _ckpt, is_best=False):
        """Store states of network and training procedure."""
        # get checkpoint file name
        if is_best:
            ckpt_name = 'best.ckpt'
        else:
            ckpt_id   = str(_ckpt['epoch']).rjust(4, '0')
            ckpt_name = 'epoch'+ckpt_id+'.ckpt'
        ckpt_file = os.path.join(self.save_dir, ckpt_name)
        torch.save(_ckpt, ckpt_file)
        # print checkpoint message
        if self.verbose:
            print('Checkpoint stored at: \t{}'.format(ckpt_file))

    def load_status(self, _ckpt):
        meter_module = importlib.import_module('quantlab.' + self.problem + '.postprocess')
        meter        = getattr(meter_module, 'Meter')()
        if _ckpt is None:
            self.i_epoch     = 1
            self.best_metric = meter.start_metric
        else:
            self.i_epoch     = _ckpt['epoch'] + 1
            self.best_metric = _ckpt['best_metric']
        if self.verbose and self.best_metric != meter.start_metric:
            meter.print_metric(self.best_metric)
        return meter
