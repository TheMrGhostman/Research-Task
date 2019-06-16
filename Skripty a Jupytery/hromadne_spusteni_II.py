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

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

path = os.getcwd()
path = path.replace(path.split('/')[-1],'')

#valid_path_GB = path + 'Tabulky a výsledky/Boosted Trees/GradientBoost_4stavy/FirstDataset/'
#valid_path_ADA = path + 'Tabulky a výsledky/Boosted Trees/AdaBoost_4stavy/FirstDataset/'

valid_path_GB = path + 'Tabulky a výsledky/Boosted Trees/GradientBoost_4stavy/SecondDataset/'
valid_path_ADA = path + 'Tabulky a výsledky/Boosted Trees/AdaBoost_4stavy/SecondDataset/'

#DATA = dat.load_dataset("first_dataset_mod")
#dodatek = 'MutInfo_best5_SD'
dodatek = 'F_classif_best5_SD'
DATA = dat.load_dataset("second_dataset_mod")


CONFIG_all = {'H_alpha': True,
		  '1.d SGF': True,
		  '2.d SGF': True,
		  'MM': [4, 6, 8, 10, 12, 14, 16],
		  'EMM': [4, 6, 8, 10, 12, 14, 16],
		  'MV': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}
"""
15. best 
 ['MM 8' 'MM 10' 'MM 12' 'MM 14' 'MM 16' 
 'EMM 8' 'EMM 10' 'EMM 12' 'EMM 14' 'EMM 16' 
 'MV 12' 'MV 13' 'MV 14' 'MV 15' 'MV 16']
10. best 
 ['MM 10' 'MM 12' 'MM 14' 
 'EMM 10' 'EMM 12' 'EMM 14' 
 'MV 13' 'MV 14' 'MV 15' 'MV 16']
5. best 
 ['MM 10' 'MV 13' 'MV 14' 'MV 15' 'MV 16']
"""

CONFIG_MI_15 = {'H_alpha': False,
			   '1.d SGF': False,
			   '2.d SGF': False,
			   'MM': [8,10,12,14,16],
			   'EMM': [8,10,12,14,16],
			   'MV': [12,13,14,15,16]}

CONFIG_MI_10 = {'H_alpha': False,
			   '1.d SGF': False,
			   '2.d SGF': False,
				'MM': [10,12,14],
			   'EMM': [10,12,14],
			   'MV': [13,14,15,16]}


CONFIG_MI_5 = {'H_alpha': False,
			   '1.d SGF': False,
			   '2.d SGF': False,
			   'MM': [10],
			   'MV': [13,14,15,16]}

"""
15. best 
 ['H_alpha'
  'MM 6' 'MM 8' 'MM 10' 'MM 12' 'MM 14' 'MM 16' 
  'EMM 8' 'EMM 10' 'EMM 12' 'EMM 14' 'EMM 16' 
  'MV 14' 'MV 15' 'MV 16']

10. best 
 ['MM 8' 'MM 10' 'MM 14'
  'EMM 10' 'EMM 12' 'EMM 14' 'EMM 16'
  'MV 14' 'MV 15' 'MV 16']
5. best 
 ['MM 10'
  'EMM 14' 'EMM 16'
   'MV 15' 'MV 16']

"""

CONFIG_MI_15se = {'H_alpha': True,
			   '1.d SGF': False,
			   '2.d SGF': False,
			   'MM': [6,8,10,12,14,16],
			   'EMM': [8,10,12,14,16],
			   'MV': [14,15,16]}

CONFIG_MI_10se = {'H_alpha': False,
			   '1.d SGF': False,
			   '2.d SGF': False,
				'MM': [8,10,14],
			   'EMM': [10,12,14,16],
			   'MV': [14,15,16]}


CONFIG_MI_5se = {'H_alpha': False,
			   '1.d SGF': False,
			   '2.d SGF': False,
			   'MM': [10],
			   'EMM': [14, 16],
			   'MV': [14,15,16]}


"""
15. best 
 ['1.d SGF' 
 'MM 6' 'MM 8' 'MM 10' 'MM 12' 'MM 14' 'MM 16' 
 'EMM 8' 'EMM 10' 'EMM 12' 'EMM 14' 'EMM 16' 
 'MV 14' 'MV 15' 'MV 16']

10. best 
 ['1.d SGF' 
 'MM 10' 'MM 12' 'MM 14' 'MM 16' 
 'EMM 10' 'EMM 12' 'EMM 14'  'EMM 16' 
 'MV 16']
5. best 
 ['MM 12' 'MM 14' 'MM 16' 'EMM 14' 'EMM 16']
"""

CONFIG_FC_15 = {'H_alpha': False,
			   '1.d SGF': True,
			   '2.d SGF': False,
			   'MM': [6,8,10,12,14,16],
			   'EMM': [8,10,12,14,16],
			   'MV': [14, 15, 16]}

CONFIG_FC_10 = {'H_alpha': False,
			   '1.d SGF': True,
			   '2.d SGF': False,
			   'MM': [10,12,14,16],
			   'EMM': [10,12,14,16],
			   'MV': [16]}


CONFIG_FC_5 = {'H_alpha': False,
			  '1.d SGF': False,
			  '2.d SGF': False,
			  'MM': [12, 14, 16],
			  'EMM': [14, 16]}

"""
15. best 
 ['1.d SGF'
  'MM 6' 'MM 8' 'MM 10' 'MM 12' 'MM 14' 'MM 16' 
  'EMM 6' 'EMM 8' 'EMM 10' 'EMM 12' 'EMM 14' 'EMM 16' 
  'MV 15' 'MV 16']
10. best 
 ['1.d SGF' 
 'MM 8' 'MM 10' 'MM 12' 'MM 14' 'MM 16'
  'EMM 10' 'EMM 12' 'EMM 14' 'EMM 16']
5. best 
 ['MM 12' 'MM 14' 'MM 16'
  'EMM 14' 'EMM 16']
"""

CONFIG_FC_15se = {'H_alpha': False,
			   '1.d SGF': True,
			   '2.d SGF': False,
			   'MM': [6,8,10,12,14,16],
			   'EMM': [6,8,10,12,14,16],
			   'MV': [15, 16]}

CONFIG_FC_10se = {'H_alpha': False,
			   '1.d SGF': True,
			   '2.d SGF': False,
			   'MM': [8,10,12,14,16],
			   'EMM': [10,12,14,16]}

# stejný jako u prvního
CONFIG_FC_5se = {'H_alpha': False,
			  '1.d SGF': False,
			  '2.d SGF': False,
			  'MM': [12, 14, 16],
			  'EMM': [14, 16]}


CONFIG_g = CONFIG_FC_5se
CONFIG_a = CONFIG_FC_5se

########################################################################################################
clf1 = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, random_state=0)
print("Počítám GB model s {} estim a {} lr".format(clf1.n_estimators, clf1.learning_rate))

tab1 = PM.fast_CF_Boosted_Trees(model=clf1, Data=DATA, configg=CONFIG_g, name="GB_100est_{}".format(dodatek), location=valid_path_GB, states=4)
print("F score average: {}".format(np.around(np.mean(tab1["F míra průměrná"].values), decimals=3)))

########################################################################################################
clf2 = GradientBoostingClassifier(n_estimators = 400, learning_rate = 0.1, random_state=0)
print("Počítám GB model s {} estim a {} lr".format(clf2.n_estimators, clf2.learning_rate))

tab2 = PM.fast_CF_Boosted_Trees(model=clf2, Data=DATA, configg=CONFIG_g, name="GB_400est_{}".format(dodatek), location=valid_path_GB, states=4)
print("F score average: {}".format(np.around(np.mean(tab2["F míra průměrná"].values), decimals=3)))

########################################################################################################
clf3 = GradientBoostingClassifier(n_estimators = 1000, learning_rate = 0.1, random_state=0)
print("Počítám GB model s {} estim a {} lr".format(clf3.n_estimators, clf3.learning_rate))

tab3 = PM.fast_CF_Boosted_Trees(model=clf3, Data=DATA, configg=CONFIG_g, name="GB_1000est_{}".format(dodatek), location=valid_path_GB, states=4)
print("F score average: {}".format(np.around(np.mean(tab3["F míra průměrná"].values), decimals=3)))

########################################################################################################
clf4 = GradientBoostingClassifier(n_estimators = 1000, learning_rate = 0.01, random_state=0)
print("Počítám GB model s {} estim a {} lr".format(clf4.n_estimators, clf4.learning_rate))

tab4 = PM.fast_CF_Boosted_Trees(model=clf4, Data=DATA, configg=CONFIG_g, name="GB_1000est_lr_0.01_{}".format(dodatek), location=valid_path_GB, states=4)
print("F score average: {}".format(np.around(np.mean(tab4["F míra průměrná"].values), decimals=3)))




########################################################################################################
clf5 = AdaBoostClassifier(n_estimators = 100, learning_rate = 1.0, random_state=0)
print("Počítám ADA model s {} estim a {} lr".format(clf5.n_estimators, clf5.learning_rate))

tab5 = PM.fast_CF_Boosted_Trees(model=clf5, Data=DATA, configg=CONFIG_a, name="ADA_100est_lr_1_{}".format(dodatek), location=valid_path_ADA, states=4)
print("F score average: {}".format(np.around(np.mean(tab5["F míra průměrná"].values), decimals=3)))

########################################################################################################
clf6 = AdaBoostClassifier(n_estimators = 400, learning_rate = .1, random_state=0)
print("Počítám ADA model s {} estim a {} lr".format(clf6.n_estimators, clf6.learning_rate))

tab6 = PM.fast_CF_Boosted_Trees(model=clf6, Data=DATA, configg=CONFIG_a, name="ADA_400est_lr_0.1_{}".format(dodatek), location=valid_path_ADA, states=4)
print("F score average: {}".format(np.around(np.mean(tab6["F míra průměrná"].values), decimals=3)))

########################################################################################################
clf7 = AdaBoostClassifier(n_estimators = 400, learning_rate = .5, random_state=0)
print("Počítám ADA model s {} estim a {} lr".format(clf7.n_estimators, clf7.learning_rate))

tab7 = PM.fast_CF_Boosted_Trees(model=clf7, Data=DATA, configg=CONFIG_a, name="ADA_400est_lr_0.5_{}".format(dodatek), location=valid_path_ADA, states=4)
print("F score average: {}".format(np.around(np.mean(tab7["F míra průměrná"].values), decimals=3)))

########################################################################################################
clf8 = AdaBoostClassifier(n_estimators = 1000, learning_rate = .1, random_state=0)
print("Počítám ADA model s {} estim a {} lr".format(clf8.n_estimators, clf8.learning_rate))

tab8 = PM.fast_CF_Boosted_Trees(model=clf8, Data=DATA, configg=CONFIG_a, name="ADA_1000est_lr_0.1_{}".format(dodatek), location=valid_path_ADA, states=4)
print("F score average: {}".format(np.around(np.mean(tab8["F míra průměrná"].values), decimals=3)))

########################################################################################################
clf9 = AdaBoostClassifier(n_estimators = 1000, learning_rate = .01, random_state=0)
print("Počítám ADA model s {} estim a {} lr".format(clf9.n_estimators, clf9.learning_rate))

tab9 = PM.fast_CF_Boosted_Trees(model=clf9, Data=DATA, configg=CONFIG_a, name="ADA_1000est_lr_0.01_{}".format(dodatek), location=valid_path_ADA, states=4)
print("F score average: {}".format(np.around(np.mean(tab9["F míra průměrná"].values), decimals=3)))
########################################################################################################

print("Vše je hotovo!!!!!!")



