### functions used to vectorize comments ###

import numpy as np
import nltk
import re
from contractions import CONTRACTION_MAP
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
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
    ### expand contractions in the sentence, make the words lowercase and add start/end tags###
    sentence = expand_contractions(sentence)
    cleaned = nltk.word_tokenize(sentence.lower())
    cleaned.insert(0, "<SOS>")
    cleaned.append("<EOS>")

    return " ".join(cleaned)


def clean_sentences(sentences):
    ### expand contractions in the sentences and make the words lowercase ###
    for sentence in sentences:
        clean_sentence(sentence)
    
    return sentences

def tokenize(comments, replies):
    ### create a word index (word to id) and reverse word index (id to word) and pad to uniform length###
    all_sentences = comments + replies
    tokenizer = Tokenizer(filters=' ')
    tokenizer.fit_on_texts(all_sentences)
    
    tensor = tokenizer.texts_to_sequences(all_sentences)
    tensor = pad_sequences(tensor, padding='post')

    return tensor, tokenizer

