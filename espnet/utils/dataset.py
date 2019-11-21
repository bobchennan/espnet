#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""This module contains pytorch dataset and dataloader implementation for chainer training."""

import gc
import torch
import torch.utils.data


class TransformDataset(torch.utils.data.Dataset):
    """Transform Dataset for pytorch backend.

    Args:
        data: list object from make_batchset
        transfrom: transform function

    """

    def __init__(self, data, transform):
        """Init function."""
        super(TransformDataset).__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        """Len function."""
        return len(self.data)

    def __getitem__(self, idx):
        """[] operator."""
        return self.transform(self.data[idx])


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class DataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self.iterator = super().__iter__()
        self.training = kwargs['shuffle']

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        if self.training:
            while True:
                yield next(self.iterator)
        else:
            for _ in range(len(self)):
                yield next(self.iterator)


class ChainerDataLoader(object):
    """Pytorch dataloader in chainer style.

    Args:
        all args for torch.utils.data.dataloader.Dataloader

    """

    def __init__(self, **kwargs):
        """Init function."""
        self.loader = DataLoader(**kwargs)
        self.len = len(kwargs['dataset'])
        self.current_position = 0
        self.epoch = 0
        self.iter = iter(self.loader)
        self.kwargs = kwargs

    def next(self):
        """Implement next function."""
        #try:
        #    ret = next(self.iter)
        #except StopIteration:
        #    self.iter._shutdown_workers()
        #    del self.iter
        #    self.iter = None
        #    gc.collect()
        #    self.iter = iter(self.loader)
        #    return self.next()
        ret = next(self.iter)
        self.current_position += 1
        if self.current_position == self.len:
            self.epoch = self.epoch + 1
            self.current_position = 0
        return ret

    def __iter__(self):
        """Implement iter function."""
        for batch in self.loader:
            yield batch

    @property
    def epoch_detail(self):
        """Epoch_detail required by chainer."""
        return self.epoch + self.current_position / self.len

    def serialize(self, serializer):
        """Serialize and deserialize function."""
        epoch = serializer('epoch', self.epoch)
        current_position = serializer('current_position', self.current_position)
        self.epoch = epoch
        self.current_position = current_position

    def start_shuffle(self):
        """Shuffle function for sortagrad."""
        self.kwargs['shuffle'] = True
        self.loader = torch.utils.data.dataloader.DataLoader(**self.kwargs)

    def finalize(self):
        """Implement finalize function."""
        del self.loader
