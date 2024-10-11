#!/usr/bin/env python3

"""
This module contains class LSTMCell which represents
a LSTM unit"""
import numpy as np


class LSTMCell:
    """class GRUCell"""
    def __init__(self, i, h, o):
        """class constructor
        i - data dimensionality
        h - hidden state dimensionality
        o - outputs dimensionality
        Wf, bf - weights and bias for forget gate
        Wu, bu - weights and bias for update gate
        Wc, bc - for the intermediate cell state
        W0, bo - weights and bias for output gate
        Wy, hy - weights and bias for output
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """forward propagation for one time step
        x_t - shape (m, i) contains data input
        m - batch size for data
        h_prev - shape(m,h) contains previous hidden state
        c_prev - shape(m,h) contains previous cell state
        h_next - next hidden state
        c_next - next cell state
        y - output of the cell"""

        x_t_concat = np.concatenate((h_prev, x_t), axis=1)

        # forget gate
        f_t = self.sigmoid(np.dot(x_t_concat, self.Wf) + self.bf)

        # update gate
        u_t = self.sigmoid(np.dot(x_t_concat, self.Wu) + self.bu)

        # concat hidden states
        # r_h_concat = np.concatenate((r_t * h_prev, x_t), axis=1)
        c_tilde = np.tanh(np.dot(x_t_concat, self.Wc) + self.bc)

        # next cell state
        c_next = f_t * c_prev + u_t * c_tilde

        # next gates
        o_t = self.sigmoid(np.dot(x_t_concat, self.Wo) + self.bo)

        # next hidden state
        h_next = o_t * np.tanh(c_next)

        # output
        y_lin = np.dot(h_next, self.Wy) + self.by

        # output with softmax activation
        y = np.exp(y_lin) / np.sum(np.exp(y_lin), axis=1, keepdims=True)

        return h_next, c_next, y

    def sigmoid(self, x):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-x))
