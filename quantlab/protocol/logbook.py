# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import importlib
import json
import math
import os
import shutil
import sys
from tensorboardX import SummaryWriter
import torch


# length of decimal literal identifying experiments
_exp_align_ = 3
# length of decimal literal identifying checkpoints
_ckpt_align_ = 4


class Logbook(object):

    def __init__(self, problem, topology, exp_id, load, verbose=True):
        """Experiment management abstraction.

        The logbook registers the information needed by the *lab daemons* to
        instantiate individual and treatment, plus the status of the current
        experiment.

        Args:
            problem (str): The data set name.
            topology (str): The network topology used to solve the data set.
            exp_id (str): The decimal literal identifying the experiment.
            load (str): Which checkpoint to load.
                It can be the checkpoint corresponding to best metric epoch
                (``best``), the last checkpoint (``last``), or the checkpoint
                of an epoch specified by the user (a decimal literal).
            verbose (bool): Whether status messages should be printed.

        Attributes:
            problem (str): The data set name.
            topology (str): The network topology used to solve the data set.
            dir_data (str): The full path to the data set folder in QuantLab.
                This path is actually a symbolic link to the real data folder,
                which should have been created on a fast storage device (e.g.,
                a Solid State Drive) to attain better training speed.
            dir_log (str): The full path to the log folder.
                This path is actually a symbolic link to the real log folder,
                which should have been created on a high-capacity storage
                device (e.g., a Hard Disk Drive) to store a large number of
                PyTorch checkpoint files and of TensorBoard event files.
            dir_save (str): The full path to the checkpoints folder.
                PyTorch checkpoints about the experiment will be stored here.
            dir_stats (str): The full path to the statistics folder.
                TensorBoard events about the experiment will be stored here.
            writer (:obj:`SummaryWriter`): The object that logs the experiment
                configuration and statistics on a TensorBoard event.
            config (:obj:`dict`): The full description of the experiment.
            ckpt (None or :obj:`dict`): The selected checkpoint from which the
                experiment should be resumed.
            best_metric (float): The best result achieved by the experiment up
                to the current iteration.
            i_epoch (int): The current iteration of the experiment.
            meter (:obj:`Meter`): The object which measures loss and data set-
                specific metrics.
            verbose (bool): Whether status messages should be printed.

        """
        self.verbose       = verbose
        self.problem       = problem
        self.topology      = topology
        self.module        = importlib.import_module('.'.join(['quantlab', self.problem, self.topology]))
        self.dir_data      = None
        self.dir_logs      = None
        self.dir_saves     = None
        self.dir_stats     = None
        self._set_folders(exp_id)
        self.writer        = SummaryWriter(log_dir=self.dir_stats)
        self.config        = None
        self._get_config()
        self.ckpt          = None
        self.i_epoch       = None
        self._load_checkpoint(load)
        self.meter         = None
        self.metrics       = {
            'train_loss':   None,
            'train_metric': None,
            'valid_loss':   None,
            'valid_metric': None
        }
        self.update_metric = None
        self.metric_period = None
        self.track_metric  = False
        self._init_measurements()

    def _set_folders(self, exp_id):
        """Get pointers to the data and experiment folders.

        Args:
            exp_id (str): The decimal literal identifying the experiment.

        """
        dir_src = sys.path[0]
        hs_file = os.path.join(dir_src, 'cfg/hard_storage.json')
        with open(hs_file, 'r') as fp:
            d = json.load(fp)
            HARD_STORAGE_DATA = os.path.join(d['data'], 'QuantLab')
            HARD_STORAGE_LOGS = os.path.join(d['logs'], 'QuantLab')
            if not os.path.isdir(HARD_STORAGE_DATA):
                raise FileNotFoundError('Storage (for data) not found: {}'.format(HARD_STORAGE_DATA))
            if not os.path.isdir(HARD_STORAGE_LOGS):
                raise FileNotFoundError('Storage (for logs) not found: {}'.format(HARD_STORAGE_LOGS))
        # get pointers to soft SHARED resources
        dir_prob = os.path.join(dir_src, self.problem)
        if not os.path.isdir(dir_prob):
            raise FileNotFoundError('Problem soft directory not found: {}'.format(dir_prob))
        dir_data = os.path.join(dir_prob, 'data')
        dir_logs = os.path.join(dir_prob, 'logs')
        # link soft pointers to hard SHARED resources
        # data
        hd_prob = os.path.join(HARD_STORAGE_DATA, self.problem)
        if not os.path.isdir(hd_prob):
            raise FileNotFoundError('Problem hard directory not found: {}'.format(hd_prob))
        hd_data = os.path.join(hd_prob, 'data')
        if not os.path.isdir(hd_data):
            raise FileNotFoundError('Data hard directory not found: {}'.format(hd_data))
        if not os.path.isdir(dir_data):
            os.symlink(hd_data, dir_data)
        # log
        hd_prob = os.path.join(HARD_STORAGE_LOGS, self.problem)
        if not os.path.isdir(hd_prob):
            os.mkdir(hd_prob)
        hd_logs = os.path.join(hd_prob, 'logs')
        if not os.path.isdir(hd_logs):
            raise FileNotFoundError('Logs hard directory not found: {}'.format(hd_logs))
        if not os.path.isdir(dir_logs):
            os.symlink(hd_logs, dir_logs)
        # get pointers to PRIVATE experiment sections
        if exp_id:
            # retrieve an existing report
            exp_id = int(exp_id)
        else:
            # create a new report
            exp_folders = [f for f in os.listdir(dir_logs) if f.startswith('exp')]
            if len(exp_folders) == 0:
                exp_id = 0
            else:
                exp_id = max(int(f.replace('exp', '')) for f in exp_folders) + 1
        dir_exp = os.path.join(dir_logs, 'exp'+str(exp_id).rjust(_exp_align_, '0'))
        if not os.path.isdir(dir_exp):
            os.mkdir(dir_exp)
        dir_saves = os.path.join(dir_exp, 'saves')
        if not os.path.isdir(dir_saves):
            os.mkdir(dir_saves)
        dir_stats = os.path.join(dir_exp, 'stats')
        if not os.path.isdir(dir_stats):
            os.mkdir(dir_stats)
        # assign pointers
        self.dir_data  = dir_data
        self.dir_logs  = dir_logs
        self.dir_saves = dir_saves
        self.dir_stats = dir_stats
        if self.verbose:
            # print setup message
            message  = 'EXPERIMENT LOGBOOK\n'
            message += 'Problem:               {}\n'.format(self.problem)
            message += 'Network topology:      {}\n'.format(self.topology)
            message += 'Data directory:        {}\n'.format(self.dir_data)
            message += 'Log directory:         {}\n'.format(self.dir_logs)
            message += 'Checkpoints directory: {}\n'.format(self.dir_saves)
            message += 'Statistics directory:  {}\n'.format(self.dir_stats)

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

    def _get_config(self):
        dir_prob = os.path.dirname(self.dir_data)
        dir_exp  = os.path.dirname(self.dir_saves)
        private_config_file = os.path.join(dir_exp, 'config.json')
        if not os.path.isfile(private_config_file):
            # no configuration in the experiment folder
            shared_config_file = os.path.join(dir_prob, 'config.json')
            if not os.path.isfile(shared_config_file):
                raise FileNotFoundError('Configuration file not found: {}'.format(shared_config_file))
            shutil.copyfile(shared_config_file, private_config_file)
            # write configuration to TensorBoard event file
            with open(private_config_file) as fp:
                d = json.load(fp)
            for k, v in d.items():
                self.writer.add_text(str(k), str(v))
        with open(private_config_file) as fp:
            self.config = json.load(fp)

    def _load_checkpoint(self, load):
        ckpt_list = os.listdir(self.dir_saves)
        if len(ckpt_list) != 0:
            # some checkpoint was found
            if load == 'best':
                ckpt_file = os.path.join(self.dir_saves, 'best.ckpt')
            elif load == 'last':
                ckpt_file = max([os.path.join(self.dir_saves, f) for f in ckpt_list], key=os.path.getctime)
            else:
                ckpt_id = str(load).rjust(_ckpt_align_, '0')
                ckpt_name = 'epoch' + ckpt_id + '.ckpt'
                ckpt_file = os.path.join(self.dir_saves, ckpt_name)
            if self.verbose:
                print('Loading checkpoint: {}'.format(ckpt_file))
            ckpt = torch.load(ckpt_file)
            self.i_epoch = ckpt['treat']['i_epoch']
        else:
            # no checkpoints was found
            ckpt = None
            self.i_epoch = 0
        self.ckpt = ckpt

    def _init_measurements(self):
        meter_module = importlib.import_module('.'.join(['quantlab', self.problem, 'utils', 'meter']))
        self.meter   = getattr(meter_module, 'Meter')(self.module.postprocess_pr, self.module.postprocess_gt)
        if self.ckpt:
            self.metrics.update(self.ckpt['protocol']['metrics'])
        else:
            self.metrics['train_loss']   = math.inf
            self.metrics['train_metric'] = self.meter.start_metric
            self.metrics['valid_loss']   = math.inf
            self.metrics['valid_metric'] = self.meter.start_metric
        self.update_metric = self.config['protocol']['update_metric']
        self.metric_period = self.config['protocol']['metric_period']

    def is_better(self, stats):
        if self.update_metric.endswith('loss'):
            # loss has decreased
            is_best = stats[self.update_metric] < self.metrics[self.udpate_metric]
        else:
            # problem main metric has improved
            is_best = self.meter.is_better(stats[self.update_metric], self.metrics[self.update_metric])
        if is_best:
            self.metrics.update(stats)
        return is_best

    def store_checkpoint(self, ckpt, is_best=False):
        """Store states of network and training procedure."""
        if is_best:
            ckpt_name = 'best.ckpt'
        else:
            ckpt_id   = str(ckpt['treat']['i_epoch']).rjust(_ckpt_align_, '0')
            ckpt_name = 'epoch'+ckpt_id+'.ckpt'
        ckpt_file = os.path.join(self.dir_saves, ckpt_name)
        torch.save(ckpt, ckpt_file)
        if self.verbose:
            print('Checkpoint stored: {}'.format(ckpt_file))

    def start_epoch(self):
        self.i_epoch += 1
        self.track_metric = (self.i_epoch % self.metric_period == 0) if self.metric_period else False
