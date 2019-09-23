# Copyright (c) 2019 UniMoRe, Matteo Spallanzani
# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import os
import torch

from quantlab.protocol.logbook import _exp_align_, _ckpt_align_


def load_pretrained(logbook, net):
    
    #get path to pretrained network
    pre_config = logbook.config['indiv']['net']['pretrained']
    if isinstance(pre_config['file'], str):
        ckpt_file = os.path.join(os.path.dirname(logbook.dir_logs), logbook.topology, 'pretrained', pre_config['file'])
        if not os.path.exists(ckpt_file):
            ckpt_file = pre_config['file']
    elif isinstance(pre_config['file'], dict):
        dir_exp = 'exp' + str(pre_config['file']['exp_id']).rjust(_exp_align_, '0')
        epoch_str = str(pre_config['file']['epoch'])
        if epoch_str.isnumeric():
            ckpt_id = epoch_str.rjust(_ckpt_align_, '0')
            ckpt_name = 'epoch' + ckpt_id + '.ckpt'
        else: 
            #e.g. for 'best', 'last'
            ckpt_name = epoch_str + '.ckpt'
        ckpt_file = os.path.join(logbook.dir_logs, dir_exp, 'saves', ckpt_name)
    if logbook.verbose:
        print('Loading checkpoint: {}'.format(ckpt_file))
        
    #load network params
    net_dict = net.state_dict()
    pretrained_dict = torch.load(ckpt_file)['indiv']['net']
    if 'parameters' in pre_config.keys():
        #load selected parameters
        parameters = []
        for group_name in pre_config['parameters']:
            parameters += [k for k in pretrained_dict.keys() if k.startswith(group_name) and not k.endswith('num_batches_tracked')]
        net_dict.update({k: v for k, v in pretrained_dict.items() if k in parameters})
    else:
        #load all parameters if not specified
        net_dict = pretrained_dict 
        
    missing_keys, unexpected_keys = net.load_state_dict(net_dict, strict=False)
    
    #report differences
    if len(missing_keys) > 0:
        print('WARNING: missing keys in pretrained net!')
        for k in missing_keys:
            print('key: %s' % k)
    if len(unexpected_keys) > 0:
        print('WARNING: unexpected keys in pretrained net!')
        for k in unexpected_keys:
            print('key: %s' % k)
