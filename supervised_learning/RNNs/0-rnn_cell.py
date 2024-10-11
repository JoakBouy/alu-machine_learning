#!/usr/bin/env python3

"""
This module contains class RNNCell which represents
a cell of a simple RNN"""
import numpy as np


class RNNCell:
    """class RNNCell"""
    def __init__(self, i, h, o):
        """class constructor
        i - data dimensionality
        h - hidden state dimensionality
        0 - outputs dimensionality
        wh, bh - weights
        bh, by - biases
        wy, by - output
        wh, bh - hidden state and input data
        """

        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """forward propagation for one time step
        x_t - shape (m, i) contains data input
        m - batch size for data
        h_prev - shape(m,h) contains previous hidden state
        h_next - next hidden state
        y - output of the cell"""
        h_x_concat = np.concatenate((h_prev, x_t), axis=1)

        # hidden state
        h_next = np.tanh(np.dot(h_x_concat, self.Wh) + self.bh)

        # cell output
        y_lin = np.dot(h_next, self.Wy) + self.by

        # output with softmax activation
        y = np.exp(y_lin) / np.sum(np.exp(y_lin), axis=1, keepdims=True)

        return h_next, y
