import sys 
sys.path.append("../modules")

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import time
from copy import copy
import os

import Datasets as dat
import Feature_Engineering as FE
import Scoring as S
import Pipeline_Methods as PM

from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid

path = os.getcwd()
path = path.replace(path.split('/')[-1],'')
valid_path_SVC = path + 'Tabulky a vysledky/SVC/FirstDataset/4_stavy/'

DATA = dat.load_dataset("first_dataset_mod")

CONFIG = {'H_alpha': True,
		  '1.d SGF': True,
		  '2.d SGF': True,
		  'MM': [4, 6, 8, 10, 12, 14, 16],
		  'EMM': [4, 6, 8, 10, 12, 14, 16],
		  'MV': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}

PARAMS = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'C': [0.001, 0.01, 0.1, 1, 10, 50]},
		  {'kernel': ['sigmoid'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'C': [0.001, 0.01, 0.1, 1, 10, 50]},
		  {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 50, 100]}]


#print(len(ParameterGrid(PARAMS)))
#for i in list(ParameterGrid(PARAMS)):
#	print(i)

PM.GridSearch(estimator=SVC() , params=PARAMS,  Data=DATA, config=CONFIG, name='svc_FD_comb_', location=valid_path_SVC, states=4)

