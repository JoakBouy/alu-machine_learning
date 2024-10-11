#!/usr/bin/env python3
"""
Module with class that
creates an encoder block for a transformer
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Class to create an encoder block for a transformer
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor
        public instance attribute:
        mha: MultiHeadAttention layer
        dense_hidden: the hidden dense layer with hidden units,
        relu activation
        dense_output: the output dense layer with dm units
        layernorm1: the first layer norm layer, with epsilon=1e-6
        layernorm2: the second layer norm layer, with epsilon=1e-6
        drouput1: the first dropout layer
        dropout2: the second dropout layer
        dm [int] - dimensionality of the model
        h [int] - number of heads
        hidden [int] - number of hidden units in fully connected layer
        drop_rate [float] - dropout rate
        """

        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Calls the encoder block and returns the block's output
        x [tensor of shape (batch, input_seq_len, dm)]:
            contains the input to the encoder block
        training [boolean]:
            determines if the model is in training
        mask - applied for multi-head attention
        returns - [tensor of shape (batch, input_seq_len, dm)]:
            contains the block's output
        """
        attention_output, _ = self.mha(x, x, x, mask)
        attention_output = self.dropout1(attention_output,
                                         training=training)
        output1 = self.layernorm1(x + attention_output)

        dense_output = self.dense_hidden(output1)
        ffn_output = self.dense_output(dense_output)
        ffn_output = self.dropout2(ffn_output,
                                   training=training)
        output2 = self.layernorm2(output1 + ffn_output)

        return output2
