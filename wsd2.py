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
def main():
    accuracy_list = []
    for execution in range(1,EXECUTIONS + 1):
        checked = 0 #termini valutati correttamente
        evaluated = 0 #termini valutati
        index_evaluated = set() #insieme degli indici già valutati
        while(evaluated <= RANGE):
            while True: #fino a quando non trova una word che può essere presa in considerazione  
                index = functions.get_random_index(index_evaluated, INDEXES_NUM)
                word = functions.get_random_word(functions.get_dictionary_tag(sentences_tag[index],sentences_sem[index]))
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
        accuracy = checked/RANGE
        print("Accuracy: ",accuracy)
        accuracy_list.append(accuracy)
    print()
    print("Esecuzioni: ",EXECUTIONS)
    print("Lista delle accuratezze: ", accuracy_list)
    print("Accuratezza media: ",mean(accuracy_list))
    
main()