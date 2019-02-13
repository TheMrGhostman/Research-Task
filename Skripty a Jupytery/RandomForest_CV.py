import sys
sys.path.append("../modules")
#custom modules imports
import Classification as CL
import datasets as d
from CrossValForRFandDT import CrossVal
#regular imports
from sklearn.ensemble import RandomForestClassifier,\
AdaBoostClassifier, GradientBoostingClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
import os
import progressbar
from itertools import chain
#import warnings

#setting path for saving of validations tabs and for import os
path = os.getcwd()
path = path.replace(path.split('\\')[-1],'') # pro windows je třeba "\\" u linuxu stačí "/"
valid_path = path + "\\Validační tabulky\\After FS\\"


"""
jelikož h-alpha = true tak kalšu na přidání proměné do funkce
a make matrix ještě není upgardovane
tak zatím tohle nepoužívat

KOMBINACE = (1, 1, 1, 1, 1)
PARAMS = [[12, 14, 16], [4, 14, 16], [5, 9, 13, 14, 15, 16]]
h_alpha = True
"""
KOMBINACE= "all"
PARAMS = [[4, 6], [8, 10], [12, 14, 16], range(5,17)]

LR = 0.1
ESTIM = np.arange(26,dtype=int)*10 + np.hstack((np.array([5]),np.zeros(25,dtype=int)))

if sys.argv[1] == "first":
    DATA = d.load_dataset(name="first_dataset")
elif sys.argv[1] == "second":
    DATA = d.load_dataset(name="second_dataset")
else:
    raise ValueError("Pouze dva datasety: first nebo second!!")

TRAIN, TEST = d.PrepareCrossFold(DATA.data)

TRANS_ada = [0, 1, 2, 7, 8, 9, 10, 15, 16, 17, 21, 25, 26, 27, 28] #FS podle adaboost a medianu
TRANS_rf = [4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28] #FS podle rf a medianu

for i in range(5):
    tabulky = CrossVal(model = RandomForestClassifier(criterion="entropy", max_depth=int(sys.argv[2])),
                           estimator_list=ESTIM,
                           train_data=TRAIN,
                           test_data=TEST,
                           kombinace=KOMBINACE,
                           params=PARAMS,
                           model_nm=f"RandomForest_after_FS_{i}", #"RandomForest",
                           dataset_nm=f"{sys.argv[1]}",
                           model_name_param=True,
                           transformace=TRANS_rf)


print("Hotovo!!!")
