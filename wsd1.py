#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 09:11:02 2021

@author: santealtamura
"""

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import semcor
import functions


tokenizer = RegexpTokenizer(r'\w+')

"""Word sense disambiguation (WSD) is an open problem of
natural language processing, which comprises the process of
identifying which sense of a word (i.e. meaning) is used in any
given sentence, when the word has a number of distinct senses
(polysemy)."""


INDEXES_NUM = 1000 #è il limite, partendo da 0, di sentence che possono essere testate relative al database SemCor
RANGE = 50 #è il numero di sentence che effettivamente verranno testate            

sentences_tag = semcor.tagged_sents()
sentences = semcor.sents()
sentences_sem = semcor.tagged_sents(tag = "sem")
randomlist = functions.get_random_indexes(INDEXES_NUM,RANGE)

            
#_____________________MAIN_________________________________________________
checked = 0
evaluated = 0
for index in randomlist:
    word = functions.get_random_noun(functions.get_dictionary_tag(sentences_tag[index],sentences_sem[index]))
    new = False
    while (not word):
        new = True
        new_index = randomlist[0]
        while new_index in randomlist:
           new_index = functions.get_random_indexes(INDEXES_NUM,1)[0]
        word = functions.get_random_noun(functions.get_dictionary_tag(sentences_tag[new_index],sentences_sem[new_index]))  
    print("====================")
    if new:
        sentence = sentences[new_index]
    else:
        sentence = sentences[index]
    evaluated = evaluated + 1
    print("Parola da disambiguare: ",word)
    best_sense = str(functions.lesk_algorithm(word, ' '.join(word for word in sentence), word_type='NOUN'))
    print("Senso attribuito dall'algoritmo di Lesk: ",best_sense)
    if new:
        target_lemma = functions.get_synset_target_for_word_in_sentence(word,sentences_sem[new_index])
    else:
        target_lemma = functions.get_synset_target_for_word_in_sentence(word,sentences_sem[index])
    print("Senso Target: ",target_lemma)
    best_sense_lemma = best_sense[8:len(best_sense)-2]
    if target_lemma:
         if best_sense_lemma in target_lemma:
             checked= checked + 1
    print("====================")
    
    
print("Checked: ", checked)
print("Evaluated: ",evaluated)
print("Accuracy: ",checked/RANGE)
#_____________________MAIN_________________________________________________
    
    
    
    
    
    
    
    
    
    