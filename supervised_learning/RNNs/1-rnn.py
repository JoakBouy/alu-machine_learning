#!/usr/bin/env python3

"""
This module contains class RNNCell which represents
a cell of a simple RNN"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """forward propagation for a simple RNN
    rnn_cell = instance of a RNNCell
    X - data used - shape(t, m, i)
        t - max time steps
        m - batch size
        i data dimensionality
    h_0 - initial hidden state shape(m,h)
        h - hidden state dimensionality
    return H, Y
        H - all hidden states
        Y - all outputs"""

    t, m, i = X.shape
    _, h = h_0.shape
    o = rnn_cell.Wy.shape[1]

    # initialize hidden state
    H = np.zeros((t+1, m, h))

    # initialize output
    Y = np.zeros((t, m, o))

    # initial hidden state
    H[0] = h_0

    for step in range(t):
        x_t = X[step]
        h_prev = H[step]
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H[step + 1] = h_next
        Y[step] = y

    return H, Y
