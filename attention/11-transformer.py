#!/usr/bin/env python3
"""
Module with a class that
creates a transformer network
"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.layers.Layer):
    """
    Class to create the transformer network
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        super(Transformer, self).__init__()
        """
        Class constructor

        N [int]:
            represents the number of blocks in the encoder and decoder
        dm [int]:
            represents the dimensionality of the model
        h [int]:
            represents the number of heads
        hidden [int]:
            represents the number of hidden units in fully connected layer
        input_vocab [int]:
            represents the size of the input vocabulary
        target_vocab [int]:
            represents the size of the target vocabulary
        max_seq_input [int]:
            represents the maximum sequence length possible for input
        max_seq_target [int]:
            represents the maximum sequence length possible for target
        drop_rate [float]:
            the dropout rate

        sets the public instance attributes:
            encoder: the encoder layer
            decoder: the decoder layer
            linear: the Dense layer with target_vocab units
        """
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate)
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """
        Calls the transformer network and returns the transformer output

        inputs [tensor of shape (batch, input_seq_len)]:
            contains the inputs
        target [tensor of shape (batch, target_seq_len)]:
            contains the target
        training [boolean]:
            determines if the model is in training
        encoder_mask:
            padding mask to be applied to the encoder
        look_ahead_mask:
            look ahead mask to be applied to the decoder
        decoder_mask:
            padding mask to be applied to the decoder

        returns:
            [tensor of shape (batch, target_seq_len, target_vocab)]:
                contains the transformer output
        """
        encoder_output = self.encoder(inputs, training, encoder_mask)
        decoder_output = self.decoder(target, encoder_output, training,
                                      look_ahead_mask, decoder_mask)
        final_output = self.linear(decoder_output)
        return final_output
