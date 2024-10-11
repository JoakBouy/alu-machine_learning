#!/usr/bin/env python3

"""
This module contains class BidirectionalCell which represents
a Bidirectional cell of an RNN"""
import numpy as np


class BidirectionalCell:
    """class GRUCell"""
    def __init__(self, i, h, o):
        """class constructor
        i - data dimensionality
        h - hidden state dimensionality
        o - outputs dimensionality
        Whf, bhf - weights and bias for hidden states
            in forward direction
        Whb, bhb - weights and bias for hidden states
            in backward direction
        Wy, hy - weights and bias for output
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """calculates hidden state in forward direction
        for one time step
        x_t - cell data input shape(m, i)
            m - batch size
        h_prev - previous hidden state shape(m, h)
        returns h_next
        h_next - next hidden state
        """
        # concatenate
        h_x_concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(h_x_concat, self.Whf) + self.bhf)
        return h_next
