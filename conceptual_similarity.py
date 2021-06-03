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

#massima profondità intesa come il massimo dei massimi percorsi di iperonimi di tutti i sensi presenti in wordnet
max_depth = max(max(len(hyp_path) for hyp_path in all_hypernym_paths(ss)) for ss in wn.all_synsets())

def similarity(word1, word2, similarity_function):
    max_sim = 0
    for s1 in wn.synsets(word1):
        for s2 in wn.synsets(word2):
           sim = similarity_function(s1,s2)
           if sim > max_sim:
               max_sim = sim
    return max_sim

#------------ SIMILARITY -------------#

def wu_palmer_similarity(sense1, sense2):
    numerator = 2 * depth(LC_subsumer(sense1,sense2))
    denominator = depth(sense1) + depth(sense2)
    return numerator/denominator

def shortest_path_similarity(sense1, sense2):
    return ((2 * max_depth) - shortest_path(sense1, sense2))

def leakcock_chodorow(sense1, sense2):
    numerator = (shortest_path(sense1, sense2) + 1)
    denominator = (2 * max_depth) + 1
    return -math.log(numerator/denominator)

#------------ SIMILARITY -------------#
#restituisce un dizionario di iperonimi del senso in input con la relativa profondità
#rispetto al senso in input
#esempio per il senso 'home.n.01':
"""{Synset('home.n.01'): 0, Synset('residence.n.01'): 1, Synset('address.n.02'): 2, 
Synset('geographic_point.n.01'): 3, Synset('point.n.02'): 4, Synset('location.n.01'): 5, 
Synset('object.n.01'): 6, Synset('physical_entity.n.01'): 7, Synset('entity.n.01'): 8}"""
def hypernym_paths(sense):
    queue = deque([(sense, 0)])
    path = {}

    while queue:
        s, depth = queue.popleft()
        if s in path:
            continue
        path[s] = depth

        depth += 1
        queue.extend((hyp, depth) for hyp in s.hypernyms())

    return path


#restituisce il percorso più breve tra due sensi di wordnet
def shortest_path(sense1,sense2):
    if sense1 == sense2:
        return 0
    
    dict1 = hypernym_paths(sense1) #dizionario iperonimi del sense1
    dict2 = hypernym_paths(sense2) #dizionario iperonimi del sense2
    
    min_dist = float("inf") #distanza minima inizialmente uguale a +infinito
    #per ogni senso del dizionario degli iperonimi di sense1
    #cerca di trovare il match con un senso del secondo dizionario.
    #la distanza minima sarà la somma minima delle profondità dei sensi matchati
    #rispetto al senso di partenza (sense1, sense2)
    for ss, dist1 in dict1.items():
        dist2 = dict2.get(ss) 
        if not dist2:
            dist2 = float("inf")
        min_dist = min(min_dist, dist1 + dist2)
    return float("inf") if math.isinf(min_dist) else min_dist
"""
#Restituisce il lowest common ancestor di due sensi, cioè l'iperonimo comune
#più vicino ai due sensi. se ne ho più di uno prendo quello più profondo, quindi meno generale
#listahyperonymsens: lista di liste, ogni elemento rappresenta la lista degli iperonimi della lista precendente
def LC_subsumer(sense1, sense2):
    senselist1 = [sense1]
    senselist2 = [sense2]
    listahyperonymssens1 = [senselist1] 
    listahyperonymssens2 = [senselist2]
    while(True):  
        if senselist1 == senselist2 == []: #quando sono entrambi vuoti vuol dire che abbiamo raggiunto il limite
            break #terminiamo il ciclo
        senselist1 = hypernyms_list(senselist1)
        senselist2 = hypernyms_list(senselist2)
        listahyperonymssens1.append(senselist1)
        listahyperonymssens2.append(senselist2)
    for list1 in listahyperonymssens1:
        for list2 in listahyperonymssens2:
            intersection = list(set(list1) & set(list2))
            if intersection:
                deepest = intersection[0]
                for sense in intersection:
                    if depth(sense) < depth(deepest):
                        deepest = sense
                return deepest #restituisce il senso con profondità maggiore rispetto alla radice
"""

#restituisce la profondità di un senso in wordnet
#la profondità è vista come il massimo percorso dal senso alla radice
#cioè il path più lungo di tutti i path degli iperonimi del senso          
def depth(sense):
    if not sense:
        return 0
    return max([len(path) for path in all_hypernym_paths(sense)])

#Restituisce il lowest common ancestor di due sensi, cioè l'iperonimo comune
#più vicino ai due sensi. se ne ho più di uno prendo quello più profondo,
#quindi meno generale
def LC_subsumer(sense1, sense2):
    if sense1 == sense2:
        return sense1
    
    dict1 = hypernym_paths(sense1) #dizionario iperonimi del sense1
    dict2 = hypernym_paths(sense2) #dizionario iperonimi del sense2
    
    min_dist = float("inf")
    candidates = []
    for ss, dist1 in dict1.items():
        dist2 = dict2.get(ss) 
        if not dist2:
            dist2 = float("inf")
        else:
            if (dist1 + dist2) <= min_dist:
                min_dist = (dist1 + dist2)
                candidates.append(ss) 
    if candidates:
        deepest = candidates[0]
        for sense in candidates:
            if depth(sense) < depth(deepest):
                deepest = sense
        return deepest #restituisce il senso con profondità maggiore rispetto alla radice
                
#restituisce una lista di iperonimi per una lista di sensi
def hypernyms_list(senselist):
    lista = []
    for sense in senselist:
        for sense_hyp in sense.hypernyms():
            lista.append(sense_hyp)
    return lista


#Classe enum con la quale specifichiamo il tipo di similarità utilizzata
class Similarity_Type(enum.Enum):
    wu_palmer_similarity = 1
    shortest_path = 2
    leakcock_chodorow = 3

similarity_type = Similarity_Type.leakcock_chodorow


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
            if similarity_type.value == 1:
                sim = similarity(row[0], row[1], wu_palmer_similarity)
            elif similarity_type.value == 2:
                sim = similarity(row[0], row[1], shortest_path_similarity)
            elif similarity_type.value == 3:
                sim = similarity(row[0], row[1], leakcock_chodorow)
            print("Similarity: ",sim)
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
    Restituisce una matrice M = nxn quadrata dove n = 2 (numero di inoput) dove M (i,j) è
    la correlazione tra la variabile casuale i e j. Poichè la correlazione
    tra una variabile e se stessa è 1, tutte le voci diagonali (i,i) sono uguali a 1
    
"""

print("Pearson Correlation: ",np.corrcoef(assignments, targets))


"""
SPEARMAN
    Misura non parametrica della monotonicità della relazione
    tra due dataset. Varia da +1 a -1 con 0 che implica la non correlazione
    1 indica che se x cresce y cresce
    -1 indica che se x cresce y descresce
"""

print()
print("Spearman Correlation: ",sp.stats.spearmanr(assignments, targets))
#_____________________MAIN_________________________________________________




