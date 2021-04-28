#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:58:29 2021

@author: santealtamura
"""
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import semcor
import functions
from statistics import mean


tokenizer = RegexpTokenizer(r'\w+')


EXECUTIONS = 10 #numero massimo di esecuzioni
INDEXES_NUM = 7000 #è il limite, partendo da 0, di sentence che possono essere testate relative al database SemCor
RANGE = 50 #è il numero di sentence che effettivamente verranno testate in ogni esecuzione        

sentences_tag = semcor.tagged_sents()
sentences = semcor.sents()
sentences_sem = semcor.tagged_sents(tag = "sem")

#_____________________MAIN_________________________________________________
accuracy_list = []
for execution in range(1,EXECUTIONS + 1):
    randomlist = functions.get_random_indexes(INDEXES_NUM,RANGE)
    checked = 0 
    evaluated = 0
    print("Esecuzione n. ", execution)
    for index in randomlist:
        word = functions.get_random_word(functions.get_dictionary_tag(sentences_tag[index],sentences_sem[index]))
        new = False
        while (not word): #ci permette di scegliere una parola valida che abbia un synset target e un almeno un synset in wordnet
            new = True
            new_index = randomlist[0]
            while new_index in randomlist:
               new_index = functions.get_random_indexes(INDEXES_NUM,1)[0]
            word = functions.get_random_word(functions.get_dictionary_tag(sentences_tag[new_index],sentences_sem[new_index]))  
        print("====================")
        if new:
            sentence = sentences[new_index]
        else:
            sentence = sentences[index]
        print("Frase: ",sentence)
        #sentence_words = functions.remove_punctuation(sentence)
        sentence_words = functions.remove_punctuation(functions.tokenize_sentence(' '.join(word for word in sentence)))
        evaluated = evaluated + 1
        print("Parola da disambiguare: ",word)
        best_sense = str(functions.lesk_algorithm(word, sentence_words))
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
    accuracy = checked/RANGE
    print("Accuracy: ",accuracy)
    accuracy_list.append(accuracy)

print()
print("Esecuzioni: ",EXECUTIONS)
print("Lista delle accuratezze: ", accuracy_list)
print("Accuratezza media: ",mean(accuracy_list))
#_____________________MAIN_________________________________________________


