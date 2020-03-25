#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Unility funcitons for Transformer."""

import numpy as np
import torch


def add_sos_eos(ys_pad, sos, eos, ignore_id):
    """Add <sos> and <eos> labels.

    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int sos: index of <sos>
    :param int eos: index of <eeos>
    :param int ignore_id: index of padding
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    """
    from espnet.nets.pytorch_backend.nets_utils import pad_list
    _sos = ys_pad.new([sos])
    _eos = ys_pad.new([eos])
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


def mask_predict(ys_pad, mask_token, eos, ignore_id, training=True):
    from espnet.nets.pytorch_backend.nets_utils import pad_list
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    _eos = ys_pad.new([eos])
    for i in range(len(ys)):
        ys[i] = torch.cat([ys[i], _eos], dim=0)
    ys_out = [y.clone() * 0 + ignore_id for y in ys]
    ys_in = [y.clone() for y in ys]
    for i in range(len(ys)):
        num_samples = np.random.randint(1, len(ys[i]) + 1)
        if training:
            idx = np.random.choice(len(ys[i]), num_samples)
        else:
            idx = np.arange(len(ys[i]))
        ys_in[i][idx] = mask_token
        ys_out[i][idx] = ys[i][idx]
    return pad_list(ys_in, mask_token), pad_list(ys_out, ignore_id)


def factorize_predict(ys_pad, mask_token, eos, ignore_id, previous_result=None):
    from espnet.nets.pytorch_backend.nets_utils import pad_list
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    _eos = ys_pad.new([eos])
    for i in range(len(ys)):
        ys[i] = torch.cat([ys[i], _eos], dim=0)
    if previous_result is None:
        ys_out = [y.clone() for y in ys]
        ys_in = [y.clone() * 0 + mask_token for y in ys]
        return pad_list(ys_in, mask_token), pad_list(ys_out, ignore_id)
    else:
        ys_in = [y.clone() * 0 + mask_token for y in ys]
        mask1 = [y.clone() * 0 for y in ys]
        mask2 = [y.clone() * 0 + 1 for y in ys]
        #previous_result = previous_result.detach().argmax(dim=-1)
        #for i in range(len(ys)):
        #    num_samples = np.random.randint(1, len(ys[i]) + 1)
        #    idx = torch.argsort(previous_result[i, :len(ys[i])], descending=True)[:num_samples]
        for i in range(len(ys)):
            num_samples = np.random.randint(1, len(ys[i]) + 1)
            probs = torch.gather(previous_result[i, :len(ys[i])],1,ys[i].unsqueeze(-1)).squeeze(-1)
            idx = torch.argsort(probs, descending=True)[:num_samples]
            ys_in[i][idx] = ys[i][idx]
            mask1[i][idx] = 1
            mask2[i][idx] = 0
        return pad_list(ys_in, mask_token), (pad_list(mask1, 0), pad_list(mask2, 0))


def factorize_predict_v2(ys_pad, mask_token, eos, ignore_id, previous_result=None, saving=None):
    from espnet.nets.pytorch_backend.nets_utils import pad_list
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    _eos = ys_pad.new([eos])
    for i in range(len(ys)):
        ys[i] = torch.cat([ys[i], _eos], dim=0)
    if previous_result is None:
        ys_out = [y.clone() for y in ys]
        ys_in, ys_out = [], []
        saving = []
        for i in range(len(ys)):
            if len(ys[i]) > 2:
                num_samples = np.random.randint(1, len(ys[i]) - 1)
            else:
                num_samples = len(ys[i])
            indices = np.arange(len(ys[i]))
            np.random.shuffle(indices)

            idx = indices[:num_samples] # not provided
            yin = ys[i].clone()
            yin[idx] = mask_token
            ys_in.append(yin)

            idx = indices[num_samples:] # provided
            yout = ys[i].clone()
            yout[idx] = ignore_id
            ys_out.append(yout)
            saving.append(idx)
        return pad_list(ys_in, mask_token), pad_list(ys_out, ignore_id), saving
    else:
        ys_in = [y.clone() * 0 + mask_token for y in ys]
        mask1 = [y.clone() * 0 for y in ys]
        mask2 = [y.clone() * 0 + 1 for y in ys]
        #previous_result = previous_result.detach().argmax(dim=-1)
        #for i in range(len(ys)):
        #    num_samples = np.random.randint(1, len(ys[i]) + 1)
        #    idx = torch.argsort(previous_result[i, :len(ys[i])], descending=True)[:num_samples]
        for i in range(len(ys)):
            ys_in[i][saving[i]] = ys[i][saving[i]]

            num_samples = np.random.randint(1, len(ys[i]) + 1 - len(saving[i]))
            probs = torch.gather(previous_result[i, :len(ys[i])],1,ys[i].unsqueeze(-1)).squeeze(-1)
            probs[saving[i]] = -1e15
            idx = torch.argsort(probs, descending=True)[:num_samples]
            ys_in[i][idx] = ys[i][idx]
            mask1[i][idx] = 1
            mask2[i][idx] = 0
        return pad_list(ys_in, mask_token), (pad_list(mask1, 0), pad_list(mask2, 0))


#from add_sos_eos import factorize_predict_v2
#import torch
#ys_pad=torch.randint(0, 9, (1,10)) * 0 + 5
#mask_token=9
#eos=0
#ignore_id=-1
#ret1=factorize_predict_v2(ys_pad, mask_token, eos, ignore_id)
#previous_result=torch.rand(1,11,10)
#ret2=factorize_predict_v2(ys_pad, mask_token, eos, ignore_id, previous_result)
