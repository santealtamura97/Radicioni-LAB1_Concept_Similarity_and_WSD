#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:01:55 2021

@author: santealtamura
"""

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn
import random
import string

tokenizer = RegexpTokenizer(r'\w+')

"""Word sense disambiguation (WSD) is an open problem of
natural language processing, which comprises the process of
identifying which sense of a word (i.e. meaning) is used in any
given sentence, when the word has a number of distinct senses
(polysemy)."""


def lesk_algorithm(word, sentence_words):
    best_sense = wn.synsets(word)[0]
    max_overlap = 0
    context = remove_stopwords(sentence_words)
    for sense in wn.synsets(word):
        signature = remove_stopwords(get_signature(sense))
        overlap = len(list(set(signature) & set(context))) #overlap
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
    return best_sense
            
def get_signature(sense):
    signature = []
    for word in tokenizer.tokenize(sense.definition()): #tokenizzo la definizione del synset
        signature.append(word)
    for example in sense.examples(): #tokenizzo ogni esempio del synset
        for word in tokenizer.tokenize(example):
            signature.append(word)
    return signature #la signature conterrà tutte le parole presenti nella definizione del senso e negli esempi

#Rimuove le stopwords da una lista di parola
def remove_stopwords(words_list):
    stopwords_list = get_stopwords()
    for word in words_list:
        if word in stopwords_list:
            words_list.remove(word)
    return words_list

#Restituisce la l'insieme di stopwords dal file delle stopwords
def get_stopwords():
    stopwords = open("utils/stop_words_FULL.txt", "r")
    stopwords_list = []
    for word in stopwords:
        stopwords_list.append(word.replace('\n', ''))
    stopwords.close()
    return stopwords_list

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

#Rimuove la punteggiatura da una lista di parole
def remove_punctuation(words_list):
    new_words_list = []
    for word in words_list:
        temp = word
        if not temp.strip(string.punctuation) == "":
            new_word = word.lower()
            new_words_list.append(new_word)
    return new_words_list

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