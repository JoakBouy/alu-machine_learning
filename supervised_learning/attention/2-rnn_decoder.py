#!/usr/bin/env python3

"""
This module contains a class RNNEncoder
that encode machine translation"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """class RNNEncoder"""

    def __init__(self, vocab, embedding, units, batch):
        super(RNNDecoder, self).__init__()
        """class contructor
        vocab - int size of input vocabulary
        embedding - dim of embedding vector
        units -  no. of hidden units in RNN cell
        batch - batch size int"""
        self.vocab = vocab
        self.embed = embedding
        self.units = units
        self.batch = batch
        self.embedding = tf.keras.layers.Embedding(self.vocab, self.embed)
        self.gru = tf.keras.layers.GRU(self.units,
                                       kernel_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(units=vocab)

    def call(self, x, s_prev, hidden_states):
        """
        x - tensor - (batch, input_seq_len)
        initial - tensor-(batch, units) - intial hidden state
        """
        units = s_prev.get_shape().as_list()[1]
        attention = SelfAttention(units)
        context, weights = attention(s_prev, hidden_states)
        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)
        y, s = self.gru(x)
        y = tf.reshape(y, (-1, y.shape[2]))
        y = self.F(y)
        return y, s
