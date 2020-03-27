### functions used to vectorize comments ###

import numpy as np
import nltk
import re
from contractions import CONTRACTION_MAP
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import 
from keras.layers import Embedding

w2v_model = None

def collect_comments(filename):
    ### collect the comments/replies from the files ###
    comments = []

    with open(filename, buffering=1000) as f:
        for comment in f:
            comments.append(comment)

    return comments


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    ### expands contractions in sentences ###
    # function from https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/bonus%20content/nlp%20proven%20approach/NLP%20Strategy%20I%20-%20Processing%20and%20Understanding%20Text.ipynb
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    
    return expanded_text


def clean_sentence(sentence):
    ### split the sentence into words and make them lowercase###
    sentence = expand_contractions(sentence)
    cleaned = nltk.word_tokenize(sentence.lower())

    return cleaned


def tag_sentence(sentence):
    ### add tags for the start and end of sentences ###
    sentence.insert(0, "<SOS>")
    sentence.append("<EOS>")
    return sentence


def tag_sentences(sentences):
    ### add tags for the start and end of sentences ###
    for sentence in sentences:
        sentence.insert(0, "<SOS>")
        sentence.append("<EOS>")
    
    return sentences

def create_vocab(comments):
    ### vectorize the sentences by turning each word into an integer ###
    word2idx = {}
    idx2word = {}

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(comments)
    dictionary = tokenizer.word_index

    for k, v in dictionary.items():
        word2idx[k] = v
        idx2word[v] = k

    return word2idx, idx2word, len(word2idx.keys())


def tokenize(encoder_text, decoder_text):
    ### tokenize bag of words to bag of ids ###
    tokenizer = Tokenizer
    encoder_sequences = tokenizer.texts_to_sequences(encoder_text)
    decoder_sequences = tokenizer.texts_to_sequences(decoder_text)

    return encoder_sequences, decoder_sequences


def pad_sequences(encoder_sequences, decoder_sequences):
    ### pad the encoder and decoder sequences to a uniform length ###
    encoder_input_data = pad_sequences(encoder_sequences, padding='post', truncating='post')
    decoder_input_data = pad_sequences(decoder_sequences, padding='post', truncating='post')

    return encoder_input_data, decoder_input_data, encoder_input_data.shape[0]


def get_embeddings(model_dir):
    ### get the embeddings from word2vec ###
    embeddings_index = {}
    with open(model_dir, buffering=1000) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    return embeddings_index


def create_embedding_matrix(word_index):
    ### create embedding matrix using pretrained word2vec model ###
    embedding_matrix = np.zeros((len(word_index) + 1, 100))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def create_embedding_layer(vocab_size, max_len, embedding_matrix):
    ### create embedding layer to be used with the word2vec model from Glove ###
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len, weights=[embedding_matrix], trainable=False)
    return embedding_layer


def reshape_data(decoder_input_data, max_len, vocab_size):
    ### resize the data to use with the neural network ###
    decoder_output_data = np.zeros((len(decoder_input_data), max_lem, vocab_size), dtype='float32')

    for i, seqs in enumerate(decoder_input_data):
        for j, seq in enumerate(seqs):
            if j > 0:
                decoder_output_data[i][j] = 1.0

    print(decoder_output_data.shape)
    return decoder_output_data

