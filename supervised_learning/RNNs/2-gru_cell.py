#!/usr/bin/env python3

"""
This module contains class GRUCell which represents
a gated recurrent unit"""
import numpy as np


class GRUCell:
    """class GRUCell"""
    def __init__(self, i, h, o):
        """class constructor
        i - data dimensionality
        h - hidden state dimensionality
        o - outputs dimensionality
        Wz, bz - weights and bias for update gate
        Wr, br - weights and bias for reset gate
        Wh, bh - weights and bias for hidden state
        Wy, hy - weights and bias for output
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """forward propagation for one time step
        x_t - shape (m, i) contains data input
        m - batch size for data
        h_prev - shape(m,h) contains previous hidden state
        h_next - next hidden state
        y - output of the cell"""

        x_t_concat = np.concatenate((h_prev, x_t), axis=1)

        # update gate
        z_t = self.sigmoid(np.dot(x_t_concat, self.Wz) + self.bz)

        # reset gate
        r_t = self.sigmoid(np.dot(x_t_concat, self.Wr) + self.br)

        # concat hidden states
        r_h_concat = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.dot(r_h_concat, self.Wh) + self.bh)

        # next gates
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # output
        y_lin = np.dot(h_next, self.Wy) + self.by

        # output with softmax activation
        y = np.exp(y_lin) / np.sum(np.exp(y_lin), axis=1, keepdims=True)

        return h_next, y

    def sigmoid(self, x):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-x))
