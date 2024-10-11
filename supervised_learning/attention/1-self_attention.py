#!/usr/bin/env python3

"""
This module contains a class RNNEncoder
that encode machine translation"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """class RNNEncoder"""

    def __init__(self, units):
        super(SelfAttention, self).__init__()
        """class contructor
        units -  no. of hidden units in alignment model
        W - dense-l, units units applied to previous decoder hidden state
        U - dense layer with units units applied to encoder hidden states
        V - dense layer with 1 units applied to tanh of sum of outputs of W & U
        """

        if type(units) is not int:
            raise TypeError(
                "units must be int representing the number of hidden units")
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """
        s_prev -tensor(batch, units) - with previous decoder hidden state
        hidden_states -tensor(batch, input_seq_len, units)
        with outputs of encoder
        return context, weights
        """
        W = self.W(tf.expand_dims(s_prev, 1))
        U = self.U(hidden_states)
        V = self.V(tf.nn.tanh(W + U))
        weights = tf.nn.softmax(V, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
