# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import torch


class TransformSubset(torch.utils.data.Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole dataset.
        indices (sequence): Indices in the whole set selected for subset.
    """
    def __init__(self, dataset, indices, transform=None):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __getitem__(self, idx):
        if self.dataset.transform != self.transform:
            self.dataset.transform = self.transform
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def transform_random_split(dataset, lengths, transforms=None):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): The dataset to be split.
        lengths (sequence): Lengths of splits to be produced.
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    if transforms is None:
        transforms = [None] * len(lengths)
    indices = torch.randperm(sum(lengths))
    return [TransformSubset(dataset, indices[offset - length:offset], transform) for offset, length, transform in zip(torch._utils._accumulate(lengths), lengths, transforms)]
