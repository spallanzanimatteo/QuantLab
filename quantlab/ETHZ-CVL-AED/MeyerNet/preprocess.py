# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import torchvision as tv
import pickle
import os
import numpy as np
import torch

class PickleDictionaryNumpyDataset(tv.datasets.VisionDataset):
    """Looks for a train.pickle or test.pickle file within root. The file has 
    to contain a dictionary with classes as keys and a numpy array with the 
    data. First dimension of the numpy array is the sample index. 
    
    Args:
        root (string): Root directory path.
        train (bool, default=True): defines whether to load the train or test set. 
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        data (numpy array): All the data samples. First dim are different samples.
        targets (list): The class_index value for each image in the dataset.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super().__init__(root, transform=transform, 
             target_transform=target_transform)
        
        self.train = train  # training set or test set
        
        if self.train: 
            path = os.path.join(root, 'train.pickle')
        else: 
            path = os.path.join(root, 'test.pickle')
            
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        dataset = dataset.items()
        
        self.classes = [k for k, v in dataset] # assume: train set contains all classes
        self.classes.sort()
        self.class_to_idx = {cl: i for i, cl in enumerate(self.classes)}
        
        self.data = np.stack([v[i] for k, v in dataset for i in range(len(v))], axis=0) #np.concatenate(list(dataset.values()))
        
        self.targets = [self.class_to_idx[k] 
                        for k, v in dataset 
                        for i in range(len(v))]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = self.data[index]
        target = self.targets[index]
        
        if self.transform is not None:
            sample = self.transform(sample) # note: dimensionaility here is atypical (not 3 dims, only 2)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return torch.from_numpy(sample).float().mul(1/2**15).unsqueeze(0).contiguous(), target

    def __len__(self):
        return len(self.data)


def _get_transforms(augment):
    assert(augment == False)
    # normMean = tuple([0]*64)
    # normStddev = tuple([2**16/2]*64)

    # train_t = tv.transforms.Compose([
    #                     tv.transforms.ToTensor(),
    #                     tv.transforms.Normalize(mean=normMean, std=normStddev)])
    # valid_t = tv.transforms.Compose([
    #                     tv.transforms.ToTensor(),
    #                     tv.transforms.Normalize(mean=normMean, std=normStddev)])
    # train_t = tv.transforms.Compose([tv.transforms.ToTensor()])
    # valid_t = tv.transforms.Compose([tv.transforms.ToTensor()])
    train_t = None
    valid_t = None
    
    if not augment:
        train_t = valid_t
    transforms = {
        'training':   train_t,
        'validation': valid_t
    }
    return transforms


def load_data_sets(dir_data, data_config):
    
    augment = data_config['augment']
    
    transforms = _get_transforms(augment)
    
    trainset = PickleDictionaryNumpyDataset(dir_data, train=True, 
                                            transform=transforms['training'])
    validset = PickleDictionaryNumpyDataset(dir_data, train=False, 
                                            transform=transforms['validation'])

    return trainset, validset, None
