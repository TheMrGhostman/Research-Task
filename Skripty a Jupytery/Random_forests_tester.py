#setting path for needed modules
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

#setting path for saving of validations tabs and for import os
path = os.getcwd()
path = path.replace(path.split('\\')[-1],'') # pro windows je třeba "\\" u linuxu stačí "/"
valid_path = path + "\\Validační tabulky\\Random-Forests\\"


"""
testuju správné načtení všech knihoven
"""
#print("Všechno je načteno správně")
#print("path", path,"; ", "valid_path", valid_path)
#print("Validační tabulky\\Ranodom-Forests?:", os.listdir(valid_path))

#vkládání parametrů přes cmd
#model_name = sys.argv[0]
# sys.argv[1]

DATA = d.load_dataset(name="first_dataset")

TRAIN, TEST = d.PrepareCrossFold(DATA.data)

if sys.argv[1]=="RF":

    tabulky = CrossVal(model = RandomForestClassifier(criterion="entropy", max_depth=3),
                       estimator_list=np.linspace(5,250,3,dtype=int).tolist(),
                       train_data=TRAIN,
                       test_data=TEST,
                       kombinace="all",
                       params = [[4, 6], [8, 10], [12, 14, 16], range(5,17)],
                       model_nm="RandomForest", #"RandomForest",
                       dataset_nm="first",
                       model_name_param=True)

    print(tabulky[0].head())

if sys.argv[1]=="ada":
    tabulky_ada3 = CrossVal(model =AdaBoostClassifier(learning_rate=float(sys.argv[2])),
                   estimator_list=[int(sys.argv[3])],
                   train_data=TRAIN,
                   test_data=TEST,
                   kombinace="all",
                   params = [[4, 6], [8, 10], [12, 14, 16], range(5,17)],
                   model_nm=f"AdaBoost_nestim={int(sys.argv[3])}_lr={float(sys.argv[2])}",
                   dataset_nm="first",
                   model_name_param=False)

    print("Hotovo !!!")

if sys.argv[1]=="test":
    l = d.merge_labels(d.get_layer(TRAIN[0],2))
    print(d.shape(TRAIN[0]))
    print(np.shape(l))
    print(np.unique(l))
