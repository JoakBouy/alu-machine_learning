#!/usr/bin/env python3

"""
This module contains a function deep_rnn that
perfoms forward propagation for deep rnn"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """forward propagation for deep rnn
    rnn_cells - list of RNNCell instances of len l
    l - no. of layers
    X - data used of shape(t, m, i)
        t - time steps
        m - batch size
        i - data dimensionality
    h_0 - innitial hidden state of shape(l, m, h)
        h - hidden state dimensionality
    returns H, Y
        H - all hidden states
        Y - outputs"""

    t, m, i = X.shape
    l, _, h = h_0.shape

    # Initialize the hidden states container
    H = np.zeros((t + 1, l, m, h))

    # Initialize the output container
    Y = []

    # Set the initial hidden state
    H[0] = h_0

    for step in range(t):
        x_t = X[step]
        h_step = []

        for layer in range(l):
            h_prev = H[step, layer]
            if layer == 0:
                h_next, y = rnn_cells[layer].forward(h_prev, x_t)
            else:
                h_next, y = rnn_cells[layer].forward(h_prev, h_step[-1])

            h_step.append(h_next)
            H[step + 1, layer] = h_next

        Y.append(y)

    Y = np.stack(Y, axis=0)

    return H, Y
