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

from sklearn.feature_selection import f_classif, mutual_info_classif

path = os.getcwd()
path = path.replace(path.split('/')[-1],'')
valid_path= path + 'Tabulky a výsledky/Feature Selection/'


DATA1 = dat.load_dataset("first_dataset_mod")
DATA2 = dat.load_dataset("second_dataset_mod")


CONFIG = {'H_alpha': True,
		  '1.d SGF': True,
		  '2.d SGF': True,
		  'MM': [4, 6, 8, 10, 12, 14, 16],
		  'EMM': [4, 6, 8, 10, 12, 14, 16],
		  'MV': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}


tab = PM.FeatureScoring(data=DATA1, config=CONFIG, method=mutual_info_classif)
tab.to_csv(valid_path + "MI_FD_4stavy.csv")

tab = PM.FeatureScoring(data=DATA2, config=CONFIG, method=mutual_info_classif)
tab.to_csv(valid_path + "MI_SD_4stavy.csv")

tab = PM.FeatureScoring(data=DATA1, config=CONFIG, method=f_classif)
tab.to_csv(valid_path + "F_classif_FD_4stavy.csv")

tab = PM.FeatureScoring(data=DATA2, config=CONFIG, method=f_classif)
tab.to_csv(valid_path + "F_classif_SD_4stavy.csv")




