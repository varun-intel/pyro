from __future__ import absolute_import, division, print_function

import torch
from collections import namedtuple

from opt_einsum.parser import einsum_symbols_base


class Factor(namedtuple('Factor', ['value', 'shift'])):
    @property
    def shape(self):
        return self.value.shape

    def ndimension(self):
        return self.value.ndimension()

    @staticmethod
    def from_tensor(tensor):
        value = tensor.new_zeros(torch.Size()).expand((1,) * len(tensor.shape))
        shift = tensor
        return Factor(value, shift)

    def to_tensor(self):
        return self.value.log() + self.log_scale


def transpose(a, axes):
    value = a.value.permute(*axes)
    log_scale = a.log_scale.permute(*axes)
    return Factor(value, log_scale)


def einsum(equation, *operands):
    """
    Log-sum-exp implementation of einsum.
    """
    assert all(isinstance(f, Factor) for f in operands)
    inputs, output = equation.split('->')
    inputs = inputs.split(',')

    shifts = []
    values = []
    for dims, operand in zip(inputs, operands):
        value, shift = operand
        for i, dim in enumerate(dims):
            if dim not in output:
                shift = shift.max(i, keepdim=True)[0]
        values.append(value / shift.exp())

        # permute shift to match output
        shift = shift.squeeze()  # FIXME fails for dims of size 1
        dims = [dim for dim in dims if dim in output]
        dims = [dim for dim in output if dim not in dims] + dims
        shift = shift.reshape((1,) * (len(output) - len(shift.shape)) + shift.shape)
        if dims:
            shift = shift.permute(*(dims.index(dim) for dim in output))
        shifts.append(shift)

    value = torch.einsum(equation, values)
    return Factor(value, sum(shifts))


# Copyright (c) 2014 Daniel Smith
# This function is copied and adapted from:
# https://github.com/dgasmith/opt_einsum/blob/a6dd686/opt_einsum/backends/torch.py
def tensordot(x, y, axes=2):
    assert isinstance(x, Factor)
    assert isinstance(y, Factor)
    xnd = x.ndimension()
    ynd = y.ndimension()

    # convert int argument to (list[int], list[int])
    if isinstance(axes, int):
        axes = range(xnd - axes, xnd), range(axes)

    # convert (int, int) to (list[int], list[int])
    if isinstance(axes[0], int):
        axes = (axes[0],), axes[1]
    if isinstance(axes[1], int):
        axes = axes[0], (axes[1],)

    # initialize empty indices
    x_ix = [None] * xnd
    y_ix = [None] * ynd
    out_ix = []

    # fill in repeated indices
    available_ix = iter(einsum_symbols_base)
    for ax1, ax2 in zip(*axes):
        repeat = next(available_ix)
        x_ix[ax1] = repeat
        y_ix[ax2] = repeat

    # fill in the rest, and maintain output order
    for i in range(xnd):
        if x_ix[i] is None:
            leave = next(available_ix)
            x_ix[i] = leave
            out_ix.append(leave)
    for i in range(ynd):
        if y_ix[i] is None:
            leave = next(available_ix)
            y_ix[i] = leave
            out_ix.append(leave)

    # form full string and contract!
    einsum_str = "{},{}->{}".format(*map("".join, (x_ix, y_ix, out_ix)))
    return einsum(einsum_str, x, y)
