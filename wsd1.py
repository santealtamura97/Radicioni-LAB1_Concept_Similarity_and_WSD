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

sentences_tag = semcor.tagged_sents() #lista di liste. Ogni lista rappresenta una sentence e ad ogni parola della sentence è associato un pos tag
sentences = semcor.sents() #lista di liste di parole. Ogni lista rappresenta una sentence
sentences_sem = semcor.tagged_sents(tag = "sem") #lista di liste. Ogni lista rappresenta una sentence e ad ogni parola della sentence è associato un synset gold

 
    
#_____________________MAIN_________________________________________________
def main():
    checked = 0 #termini valutati correttamente
    evaluated = 0 #termini valutati
    index_evaluated = set() #insieme degli indici già valutati
    while(evaluated <= RANGE):
        while True: #fino a quando non trova una word che ha almeno un senso in wordnet   
            index = functions.get_random_index(index_evaluated, INDEXES_NUM)
            word = functions.get_random_noun(functions.get_dictionary_tag(sentences_tag[index],sentences_sem[index]))
            if word:
                break
        print("====================")
        sentence = sentences[index]
        print("Parola da disambiguare: ",word)
        #all'algoritmo di lesk viene dato in input la word e l'insieme dei termini che formano la frase, uniti sottoforma di stringa
        best_sense = str(functions.lesk_algorithm(word, ' '.join(word for word in sentence), word_type='NOUN'))
        print("Senso attribuito dall'algoritmo di Lesk: ",best_sense)
        target_lemma = functions.get_synset_target_for_word_in_sentence(word,sentences_sem[index])
        print("Senso Target: ",target_lemma)
        best_sense_lemma = best_sense[8:len(best_sense)-2]
        if target_lemma:
            if best_sense_lemma in target_lemma:
             checked= checked + 1
        print("====================")
        evaluated += 1
        print("====================")
    
    print("Checked: ", checked)
    print("Evaluated: ",evaluated)
    print("Accuracy: ",checked/RANGE)
    
main()   
    