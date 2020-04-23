# help from https://www.tensorflow.org/tutorials/text/nmt_with_attention#download_and_prepare_the_dataset

import tensorflow as tf
from tensorflow.keras.layers import GRU, Embedding, Layer, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy

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


class Decoder(Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc = Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)


    def call(self, x, hidden, enc_output):
        # encoder output shape = (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embeddin = (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation = (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape = (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape = (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


def loss_function(real, pred):
    optimizer = Adam()
    loss_object = sparse_categorical_crossentropy(from_logits=True, reduction='none')

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

