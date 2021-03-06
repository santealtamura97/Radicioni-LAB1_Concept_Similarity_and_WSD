#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 17:18:27 2021

@author: santealtamura
"""

from nltk.corpus import wordnet as wn
import math
import numpy as np
import scipy as sp
from collections import deque 
import enum

#Restituisce tutti i percorsi di iperonimi di un senso di wordnet
#effettua una ricerca in profondit√† ricorsiva
def all_hypernym_paths(sense):
    paths = []
    hypernyms = sense.hypernyms()
    if len(hypernyms) == 0:
        paths = [[sense]]
    for hypernym in hypernyms:
        for ancestor_list in all_hypernym_paths(hypernym):
            ancestor_list.append(sense)
            paths.append(ancestor_list)
    return paths

#massima profondit√† intesa come il massimo dei massimi percorsi di iperonimi di tutti i sensi presenti in wordnet
max_depth = max(max(len(hyp_path) for hyp_path in all_hypernym_paths(ss)) for ss in wn.all_synsets())

def similarity(word1, word2, similarity_function):
    max_sim = 0
    for s1 in wn.synsets(word1):
        for s2 in wn.synsets(word2):
           sim = similarity_function(s1,s2)
           if sim > max_sim:
               max_sim = sim
    return max_sim

#------------ MY SIMILARITY FUNCTIONS-------------#
def wu_palmer_similarity(sense1, sense2):
    numerator = 2 * depth(LC_subsumer(sense1,sense2))
    denominator = depth(sense1) + depth(sense2)
    return numerator/denominator

def leakcock_chodorow(sense1, sense2):
    numerator = (shortest_path(sense1, sense2) + 1)
    denominator = (2 * max_depth) + 1
    return -math.log(numerator/denominator)

def shortest_path_similarity(sense1, sense2):
    return ((2 * max_depth) - shortest_path(sense1, sense2))

#------------ WORDNET SIMILARITY FUNCTIONS---------#
def wn_wu_palmer_similarity(sense1, sense2):
    sim = sense1.wup_similarity(sense2)
    if sim: return sim
    else: return 0
    
def wn_leakcock_chodorow(sense1, sense2):
    try:
        sim = sense1.lch_similarity(sense2)
        if sim: return sim
        else: return 0
    except:
        return 0

#restituisce un dizionario di iperonimi del senso in input con la relativa profondit√†
#rispetto al senso in input
#le profondit√† di ogni senso sono sempre le pi√Ļ piccole distanze dal senso da cui si √® partiti
#esempio per il senso 'apple.n.01':
"""{Synset('apple.n.01'): 0, Synset('edible_fruit.n.01'): 1, Synset('pome.n.01'): 1,
 Synset('fruit.n.01'): 2, Synset('produce.n.01'): 2, Synset('reproductive_structure.n.01'): 3,
 Synset('food.n.02'): 3, Synset('plant_organ.n.01'): 4, Synset('solid.n.01'): 4, 
 Synset('plant_part.n.01'): 5, Synset('matter.n.03'): 5,
 Synset('natural_object.n.01'): 6, Synset('physical_entity.n.01'): 6, 
 Synset('whole.n.02'): 7, Synset('entity.n.01'): 7, Synset('object.n.01'): 8}"""
def min_hypernyms_depth(sense):
    sense_depth_queue = deque([(sense, 0)]) #deque garantisce efficienza nell'inserire ed eliminare elementi sia a destra che a sinistra della coda
    sense_depth_dict = {}
    while sense_depth_queue:
        s, depth = sense_depth_queue.popleft()
        if s in sense_depth_dict:
            continue
        sense_depth_dict[s] = depth
        depth += 1
        sense_depth_queue.extend((hyperonym, depth) for hyperonym in s.hypernyms()) #li mette a destra della coda
    return sense_depth_dict

#restituisce il percorso pi√Ļ breve tra due sensi di wordnet
def shortest_path(sense1,sense2):
    if sense1 == sense2:
        return 0
    
    dict1 = min_hypernyms_depth(sense1) #dizionario iperonimi del sense1
    dict2 = min_hypernyms_depth(sense2) #dizionario iperonimi del sense2
    
    min_dist = (2 * max_depth) #distanza minima inizialmente uguale alla massima lunghezza che pu√≤ avere
    #per ogni senso del dizionario degli iperonimi di sense1
    #cerca di trovare il match con un senso del secondo dizionario.
    #la distanza minima sar√† la somma minima delle profondit√† dei sensi matchati
    #rispetto al senso di partenza (sense1, sense2)
    for ss, dist1 in dict1.items():
        dist2 = dict2.get(ss) 
        if not dist2:
            dist2 = (2 * max_depth)
        min_dist = min(min_dist, dist1 + dist2)
    return (2 * max_depth) if math.isinf(min_dist) else min_dist

#Restituisce il lowest common ancestor di due sensi, cio√® l'iperonimo comune
#pi√Ļ vicino ai due sensi. se ne ho pi√Ļ di uno prendo quello pu√≤ essere il pi√Ļ profondo rispetto alla radice,
def LC_subsumer(sense1, sense2):
    if sense1 == sense2: return sense1
    
    dict1 = min_hypernyms_depth(sense1) #dizionario iperonimi minimi del sense1
    dict2 = min_hypernyms_depth(sense2) #dizionario iperonimi minimi del sense2
    
    min_dist = (2 * max_depth)
    candidates = []
    for ss, dist1 in dict1.items():
        dist2 = dict2.get(ss) 
        if not dist2:
            dist2 = (2 * max_depth)
        else:
            if (dist1 + dist2) <= min_dist:
                min_dist = (dist1 + dist2)
                candidates.append(ss) 
    if candidates:
        deepest = candidates[0]
        for sense in candidates:
            if depth(sense) < depth(deepest):
                deepest = sense
        return deepest #restituisce il senso con profondit√† maggiore rispetto alla radice

#restituisce la profondit√† di un senso in wordnet
#la profondit√† √® vista come il massimo percorso dal senso alla radice
#cio√® il path pi√Ļ lungo di tutti i path degli iperonimi del senso          
def depth(sense):
    if not sense:
        return 0
    return max([len(path) for path in all_hypernym_paths(sense)])

#Classe enum con la quale specifichiamo il tipo di similarit√† utilizzata
class Similarity_Type(enum.Enum):
    wu_palmer_similarity = 1
    shortest_path = 2
    leakcock_chodorow = 3


def main():  
    similarity_type = Similarity_Type.wu_palmer_similarity
    
    #_____________________MAIN_________________________________________________
    import csv
    with open('utils/WordSim353.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        assignments = []
        targets = []
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                print("=============================")
                print("Word1: ",row[0])
                print("Word2: ",row[1])
                sim = 0
                wn_sim = 0
                if similarity_type.value == 1:
                    sim = similarity(row[0], row[1], wu_palmer_similarity)
                    wn_sim = similarity(row[0], row[1], wn_wu_palmer_similarity)
                elif similarity_type.value == 2:
                    sim = similarity(row[0], row[1], shortest_path_similarity)
                elif similarity_type.value == 3:
                    sim = similarity(row[0], row[1], leakcock_chodorow)
                    wn_sim = similarity(row[0], row[1], wn_leakcock_chodorow)
                print("System Similarity: ",sim)
                print("WordNet Similarity: ",wn_sim)
                print("Human Similarity: ",row[2])
                print("Similarity Type: ", similarity_type.name)
                assignments.append(sim)
                targets.append(float(row[2]))
                print("=============================")
            line_count = line_count + 1
    print()
    """
    PEARSON
        +1 - Complete positive correlation
        +0.8 - Strong positive correlation
        +0.6 - Moderate positive correlation
        0 - no correlation whatsoever
        -0.6 - Moderate negative correlation
        -0.8 - Strong negative correlation
        -1 - Complete negative correlation
        Restituisce una matrice M = nxn quadrata dove n = 2 (numero di inoput) dove M (i,j) √®
        la correlazione tra la variabile casuale i e j. Poich√® la correlazione
        tra una variabile e se stessa √® 1, tutte le voci diagonali (i,i) sono uguali a 1
    """    
    print("Pearson Correlation: ",np.corrcoef(assignments, targets))
    
    """
    SPEARMAN
        Misura non parametrica della monotonicit√† della relazione
        tra due dataset. Varia da +1 a -1 con 0 che implica la non correlazione
        1 indica che se x cresce y cresce
        -1 indica che se x cresce y descresce
    """
    print()
    print("Spearman Correlation: ",sp.stats.spearmanr(assignments, targets))
    #_____________________MAIN_________________________________________________

#TESTING
main()


