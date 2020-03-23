### functions used to vectorize comments ###

import numpy as np
import nltk
import re
from contractions import CONTRACTION_MAP

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


def getVector(w):
    ### vectorize a word with Word2Vec ###
    global w2v_model
    if w in w2v_model:
        return w2v_model[w]
    else:
        return np.zeros(300)


def embed_sentence(sentence):
    ### embed each word in the sentence ###
    embedded_sentence = []
    
    for word in sentence:
        word_embedding = getVector(word)
        embedded_sentence.append(word_embedding)

    return embedded_sentence


def clean_and_embed_file(file, model):
    ### clean and embed each sentence in a file of comments ###
    global w2v_model
    w2v_model = model
    vectorized_comments = []

    comments = collect_comments(file)
    for sentence in comments:
        cleaned_sentence = clean_sentence(sentence)
        embedded_sentence = embed_sentence(cleaned_sentence)
        vectorized_comments.append(embedded_sentence)

    return vectorized_comments


def clean_and_embed_sentence(sentence, model):
    ### clean and embed a sentence ###
    global w2v_model
    w2v_model = model

    cleaned_sentence = clean_sentence(sentence)
    embedded_sentence = embed_sentence(cleaned_sentence)

    return embedded_sentence