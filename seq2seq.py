# help from https://www.tensorflow.org/tutorials/text/nmt_with_attention#download_and_prepare_the_dataset

import tensorflow as tf
from tensorflow.keras.layers import GRU, Embedding, Layer, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from preprocess import tokenizer, clean_sentence, tokenize_words
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def train_step(input, target, enc_hidden, encoder, decoder, optimizer):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(input, enc_hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([tokenizer.word_index['<sos>']] * batch_size, 1)

        # geed the target as the next input
        for t in range(1, target.shape[1]):
            # pass encoder output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(target[:, t], predictions)

            dec_input = tf.expand_dims(target[:, t], 1)

    batch_loss = (loss / int(target.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def evaluate(sentence, max_length_target, max_length_input, units):
    ### evaluate the trained model ###
    attention_plot = np.zeros((max_length_target, max_length_input))

    sentence = clean_sentence(sentence)
    input_sentence = [tokenizer.word_index[i] for i in sentence.split(' ')]
    input_tensor = tokenize_words(input_sentence, max_legth_input)

    result = ''
    hidden = [tf.zeros((1, units))]
    encoder_output, encoder_hidder = encoder(input_tensor, hidden)

    decoder_hidden = encoder_hidden
    decoder_input = tf.expand_dims([tokenizer.word_index['<sos>']], 0)

    for t in range(max_length_target):
        predictions, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden, encoder_output)

        # store the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += tokenizer.index_word[predicted_id] + ' '

        if tokenizer.index_word[predicted_id] == '<eos>':
            return result, sentence, attention_plot


        # predictied id is fed back into the model
        decoder_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


def plot_attention(attention, sentence, predicted_sentence):
    ### plot the attention weights ###
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def reply(sentence, max_length_target, max_length_input, units):
    result, sentence, attention_plot = evaluate(entence, max_length_target, max_length_input, units)

    print('Input: ' + sentence)
    print('Response: {}'.format(result))

    attention_plot = attention_plot[:len(reslut.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))

