#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:01:55 2021

@author: santealtamura
"""

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn
import random
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re

#il tokenizer utilizzato rimuove la punteggiatura
tokenizer = RegexpTokenizer(r'\w+')

"""Word sense disambiguation (WSD) is an open problem of
natural language processing, which comprises the process of
identifying which sense of a word (i.e. meaning) is used in any
given sentence, when the word has a number of distinct senses
(polysemy)."""

#Algoritmo di Lesk
def lesk_algorithm(word, sentence, word_type):
    best_sense = wn.synsets(word)[0]
    max_overlap = 0
    max_signature = []
    print("Frase: ", sentence)
    context = pre_processing(sentence)
    
    if word_type == 'NOUN':
        synsets = wn.synsets(word, pos=wn.NOUN)
    elif word_type == 'ALL':
        synsets = wn.synsets(word)
        
    for sense in synsets:
        signature = get_signature(sense)
        overlap = len(list(signature & context)) #overlap
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
            max_signature = signature
            
    print("Contesto della frase: ", context)
    print("max Signature: ", max_signature) #signature del senso con più overlap con il contesto della sentence
    return best_sense

#Signature di un senso di WordNet      
def get_signature(sense):
    signature = set()
    for word in pre_processing(sense.definition()): #tokenizzo la definizione del synset
        signature.add(word)
    for example in sense.examples(): #tokenizzo ogni esempio del synset
        for word in pre_processing(example):
            signature.add(word)
    return signature #la signature conterrà tutte le parole presenti nella definizione del senso e negli esempi

"""Funzioni di supporto"""

#il pre-processing consiste nella tokenizzazione, lemmatizzazione,
#rimozione della punteggiatura e delle stopwords di una sentence
def pre_processing(sentence):
    return set(remove_stopwords(tokenize_sentence(remove_punctuation(sentence))))

#rimuove le stowords da una lista di parole
def remove_stopwords(words_list):
    stopwords_list = get_stopwords()
    return [value for value in words_list if value not in stopwords_list]


#Tokenizza la frase in input e ne affettua anche la lemmatizzazione della sue parole
def tokenize_sentence(sentence):
    words_list = []
    lmtzr = WordNetLemmatizer()
    for tag in nltk.pos_tag(word_tokenize(sentence)):
        if (tag[1][:2] == "NN"):
            words_list.append(lmtzr.lemmatize(tag[0], pos = wn.NOUN))
        elif (tag[1][:2] == "VB"):
             words_list.append(lmtzr.lemmatize(tag[0], pos = wn.VERB))
        elif (tag[1][:2] == "RB"):
             words_list.append(lmtzr.lemmatize(tag[0], pos = wn.ADV))
        elif (tag[1][:2] == "JJ"):
             words_list.append(lmtzr.lemmatize(tag[0], pos = wn.ADJ))
    return words_list

#Restituisce la l'insieme di stopwords dal file delle stopwords
def get_stopwords():
    stopwords = open("utils/stop_words_FULL.txt", "r")
    stopwords_list = []
    for word in stopwords:
        stopwords_list.append(word.replace('\n', ''))
    stopwords.close()
    return stopwords_list

#Rimuove la punteggiatura da una sentence
#Restituisce la sentence senza punteggiature
def remove_punctuation(sentence):
    return re.sub(r'[^\w\s]','',sentence)

"""Funzioni utili"""

#Restituisce un sostantivo random presente nel dizionario associato ad una sentence
def get_random_noun(dictionary_tag):
    try:
        sentence_nouns = dictionary_tag['NN']
    except KeyError:
        return None #se non ci sono sostantivi
    noun =  random.choice(sentence_nouns)
    if len(wn.synsets(noun)) == 0:
        return None #se il sostantivo scelto non ha almeno un synset in wordnet
    return noun

#Restituisce una parola random presente nel dizionario associato ad una sentence
def get_random_word(dictionary_tag):
    keys = list(dictionary_tag.keys())
    if not keys:
        return None #se non ci sono parole "analizzabili" nel dizionario, aka se il dizionario è vuoto
    key = random.choice(keys)
    words_list = dictionary_tag[key]
    word = random.choice(words_list)
    if len(wn.synsets(word)) == 0:
        return None #se la parola scelta non ha almeno un synset in wordnet
    return word

#Verifica che una parola abbia un synset target associato
#Senza synset target associato è inutile confrontare l'output dell'algortimo di Lesk
def check_word_synset_target(word,sentence_sem):
    for w in sentence_sem:
        if (type(w) != list):
            if (w[0] == word):
                return True
    return False
    
#Crea un dizionario che ha come chiavi i tag presenti nella sentence
#e come valori liste di parole associate a quei tag
#e.g {NN: ['home','garden'], VB:['gone'], .....,}
#input: sentence_tag -> rappresenta le words della sentence con i relativi pos tag
#input: sentence_sem -> rappresenta le words della sentence con il relativo synset target associato
def get_dictionary_tag(sentence_tag,sentence_sem):
    dictionary_tag = dict()
    stopwords_list = get_stopwords()
    for word in sentence_tag:
         tag = word.label()
         word = " ".join(l for l in word)
         if check_word_synset_target(word, sentence_sem): #Non inserisco nel dizionario parole che non hanno un synset target
             w = word.lower()
             if w not in stopwords_list and tag: #Non inserisco nel dizionario stopwords o parole che non hanno un pos tag associato
                 if tag in dictionary_tag:
                     dictionary_tag[tag].append(word)
                 else:
                     dictionary_tag[tag] = [word]             
    return dictionary_tag


#Restituisce una lista di indici random di numero pari a ran e come limite indexes_num
def get_random_indexes(indexes_num, ran):
    randomlist = []
    for i in range(0,ran):    
        n = random.randint(0,indexes_num)
        while n in randomlist:
            n = random.randint(0,indexes_num)
        randomlist.append(n)
    return randomlist

#Restituisce il synset target per una parola in una sentence
def get_synset_target_for_word_in_sentence(noun,sentence):
     for word in sentence:
      if(type(word) != list):
          if (word[0] == noun):
              return str(word.label())
