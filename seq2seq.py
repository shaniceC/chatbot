# help from https://www.tensorflow.org/tutorials/text/nmt_with_attention#download_and_prepare_the_dataset

import tensorflow as tf
from tensorflow.keras.layers import GRU, Embedding, Layer, Dense
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
        return tf.zeros((self.batch_size, self.enc_unitsshape))


class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)


    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape = (batch_size, max_length, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        # attention weights shape = (batch_size, hidden_size)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context vector shape after sum = (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
