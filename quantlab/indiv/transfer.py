# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import os
import torch

from quantlab.protocol.logbook import _exp_align_, _ckpt_align_


def load_pretrained(logbook, net):
    pre_config = logbook.config['indiv']['net']['pretrained']
    if isinstance(pre_config['file'], str):
        ckpt_file = os.path.join(os.path.dirname(logbook.dir_logs), logbook.topology, 'pretrained', pre_config['file'])
    elif isinstance(pre_config['file'], dict):
        dir_exp = 'exp' + str(pre_config['file']['exp_id']).rjust(_exp_align_, '0')
        ckpt_id = str(pre_config['file']['epoch']).rjust(_ckpt_align_, '0')
        ckpt_name = 'epoch' + ckpt_id + '.ckpt'
        ckpt_file = os.path.join(logbook.dir_logs, dir_exp, 'saves', ckpt_name)
    if logbook.verbose:
        print('Loading checkpoint: {}'.format(ckpt_file))
    net_dict = net.state_dict()
    pretrained_dict = torch.load(ckpt_file)['indiv']['net']
    parameters = []
    for group_name in pre_config['parameters']:
        parameters += [k for k in pretrained_dict.keys() if k.startswith(group_name) and not k.endswith('num_batches_tracked')]
    net_dict.update({k: v for k, v in pretrained_dict.items() if k in parameters})
    net.load_state_dict(net_dict)
