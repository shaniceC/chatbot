# help from https://www.tensorflow.org/tutorials/text/nmt_with_attention#download_and_prepare_the_dataset

import tensorflow as tf
from tensorflow.keras.layers import GRU, Embedding
from tensorflow.keras.models import Model

class Encoder(Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')


    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state


    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))