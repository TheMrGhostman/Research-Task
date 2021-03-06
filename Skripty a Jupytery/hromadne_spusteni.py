import sys 
sys.path.append("../modules")

import numpy as np 
import os

import Datasets as dat
import Pipeline_Methods as PM

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

path = os.getcwd()
path = path.replace(path.split('/')[-1],'')
# valid_path_GB = path + 'Tabulky a výsledky/Boosted Trees/GradientBoost_4stavy/SecondDataset/'
# valid_path_ADA = path + 'Tabulky a výsledky/Boosted Trees/AdaBoost_4stavy/SecondDataset/'
# older
print(path)
#valid_path_GB = path + 'Tabulky a výsledky/Boosted Trees/GradientBoost AFS/Feature_importance/Second/'
#valid_path_ADA = path + 'Tabulky a výsledky/Boosted Trees/AdaBoost ASF/Feature_importance/First/'

valid_path_GB = path + 'Tabulky a výsledky/Boosted Trees/GradientBoost_4stavy/FirstDataset/MI/'
valid_path_ADA = path + 'Tabulky a výsledky/Boosted Trees/AdaBoost_4stavy/FirstDataset/MI/'

#DATA = dat.load_dataset("first_dataset")
dodatek = 'MutInfo_best10_FD'
DATA = dat.load_dataset("first_dataset_mod")


CONFIG_all = {'H_alpha': True,
		  '1.d SGF': True,
		  '2.d SGF': True,
		  'MM': [4, 6, 8, 10, 12, 14, 16],
		  'EMM': [4, 6, 8, 10, 12, 14, 16],
		  'MV': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}


CONFIG_mi_10_fd = {'H_alpha': False,
		  '1.d SGF': False,
		  '2.d SGF': False,
		  'MM': [10, 12, 14],
		  'EMM': [10, 12, 14],
		  'MV': [13, 14, 15, 16]}


CONFIG_GB_15 = {'H_alpha': True,
			   '1.d SGF': True,
			   '2.d SGF': True,
			   'MM': [4,16],
			   'EMM': [4,6,16],
			   'MV': [5,6,11,13,14,15,16]}

CONFIG_GB_10 = {'H_alpha': True,
			   '1.d SGF': True,
			   '2.d SGF': True,
			   'MM': [16],
			   'EMM': [4,16],
			   'MV': [5,13,14,16]}

CONFIG_GB_5 = {'H_alpha': True,
			   '1.d SGF': True,
			   '2.d SGF': True,
			   'MM': [16],
			   'MV': [16]}

CONFIG_GB_15se = {'H_alpha': True,
			  '1.d SGF': True,
			  '2.d SGF': True,
			  'MM': [4, 6, 16],
			  'EMM': [4, 6, 8, 16],
			  'MV': [5, 11, 14, 15, 16]}

CONFIG_GB_10se = {'H_alpha': True,
			  '1.d SGF': True,
			  '2.d SGF': True,
			  'MM': [6, 16],
			  'EMM': [4, 16],
			  'MV': [5, 15, 16]}

CONFIG_GB_5se = {'H_alpha': True,
			  '1.d SGF': True,
			  '2.d SGF': True,
			  'MM': [16],
			  'MV': [16]}


"""	
4 stavy SD

CONFIG_GB_15se = {'H_alpha': True,
			  '1.d SGF': True,
			  '2.d SGF': True,
			  'MM': [4, 6, 16],
			  'EMM': [4, 6, 8, 16],
			  'MV': [5, 13, 14, 15, 16]}


CONFIG_GB_10se = {'H_alpha': True,
			  '1.d SGF': True,
			  '2.d SGF': True,
			  'MM': [4, 6, 16],
			  'EMM': [4, 16],
			  'MV': [5, 16]}


CONFIG_GB_5se = {'H_alpha': True,
			  '1.d SGF': True,
			  '2.d SGF': True,
			  'MM': [16],
			  'MV': [16]}

CONFIG_Ada_15se = {'H_alpha': True,
			  '1.d SGF': True,
			  '2.d SGF': True,
			  'MM': [6, 14, 16],
			  'MV': [5, 6, 8, 11, 12, 13, 14, 15, 16]}

CONFIG_Ada_10se = {'H_alpha': True,
			  '1.d SGF': True,
			  '2.d SGF': True,
			  'MM': [6, 16],
			  'MV': [5, 11, 14, 15, 16]}


CONFIG_Ada_5se = {'H_alpha': True,
			  '1.d SGF': True,
			  '2.d SGF': False,
			  'MM': [6, 16],
			  'MV': [16]}
"""

CONFIG_Ada_15se = {'H_alpha': True,
			  '1.d SGF': True,
			  '2.d SGF': True,
			  'MM': [6, 14, 16],
			  'MV': [5, 6, 8, 11, 12, 13, 14, 15, 16]}

CONFIG_Ada_10se = {'H_alpha': True,
			  '1.d SGF': True,
			  '2.d SGF': True,
			  'MM': [6, 16],
			  'MV': [5, 11, 14, 15, 16]}


CONFIG_Ada_5se = {'H_alpha': True,
			  '1.d SGF': True,
			  '2.d SGF': False,
			  'MM': [6, 16],
			  'MV': [16]}

CONFIG_g = CONFIG_mi_10_fd
CONFIG_a = CONFIG_mi_10_fd

print("Začínáme!!")
"""
########################################################################################################
clf1 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0)
print("Počítám GB model s {} estim a {} lr".format(clf1.n_estimators, clf1.learning_rate))

tab1 = PM.fast_CF_Boosted_Trees(model=clf1, Data=DATA, config=CONFIG_g, name="GB_100est_{}".format(dodatek), location=valid_path_GB, states=4)
print("F score average: {}".format(np.around(np.mean(tab1["F míra průměrná"].values), decimals=3)))

########################################################################################################
clf2 = GradientBoostingClassifier(n_estimators=400, learning_rate=0.1, random_state=0)
print("Počítám GB model s {} estim a {} lr".format(clf2.n_estimators, clf2.learning_rate))

tab2 = PM.fast_CF_Boosted_Trees(model=clf2, Data=DATA, config=CONFIG_g, name="GB_400est_{}".format(dodatek), location=valid_path_GB, states=4)
print("F score average: {}".format(np.around(np.mean(tab2["F míra průměrná"].values), decimals=3)))

########################################################################################################
clf3 = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, random_state=0)
print("Počítám GB model s {} estim a {} lr".format(clf3.n_estimators, clf3.learning_rate))

tab3 = PM.fast_CF_Boosted_Trees(model=clf3, Data=DATA, config=CONFIG_g, name="GB_1000est_{}".format(dodatek), location=valid_path_GB, states=4)
print("F score average: {}".format(np.around(np.mean(tab3["F míra průměrná"].values), decimals=3)))

########################################################################################################
clf4 = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01, random_state=0)
print("Počítám GB model s {} estim a {} lr".format(clf4.n_estimators, clf4.learning_rate))

tab4 = PM.fast_CF_Boosted_Trees(model=clf4, Data=DATA, config=CONFIG_g, name="GB_1000est_lr_0.01_{}".format(dodatek), location=valid_path_GB, states=4)
print("F score average: {}".format(np.around(np.mean(tab4["F míra průměrná"].values), decimals=3)))

"""

########################################################################################################
clf5 = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=0)
print("Počítám ADA model s {} estim a {} lr".format(clf5.n_estimators, clf5.learning_rate))

tab5 = PM.fast_CF_Boosted_Trees(model=clf5, Data=DATA, config=CONFIG_a, name="ADA_100est_lr_1_{}".format(dodatek), location=valid_path_ADA, states=4)
print("F score average: {}".format(np.around(np.mean(tab5["F míra průměrná"].values), decimals=3)))

########################################################################################################
clf6 = AdaBoostClassifier(n_estimators=400, learning_rate=.1, random_state=0)
print("Počítám ADA model s {} estim a {} lr".format(clf6.n_estimators, clf6.learning_rate))

tab6 = PM.fast_CF_Boosted_Trees(model=clf6, Data=DATA, config=CONFIG_a, name="ADA_400est_lr_0.1_{}".format(dodatek), location=valid_path_ADA, states=4)
print("F score average: {}".format(np.around(np.mean(tab6["F míra průměrná"].values), decimals=3)))

########################################################################################################
clf7 = AdaBoostClassifier(n_estimators=400, learning_rate=.5, random_state=0)
print("Počítám ADA model s {} estim a {} lr".format(clf7.n_estimators, clf7.learning_rate))

tab7 = PM.fast_CF_Boosted_Trees(model=clf7, Data=DATA, config=CONFIG_a, name="ADA_400est_lr_0.5_{}".format(dodatek), location=valid_path_ADA, states=4)
print("F score average: {}".format(np.around(np.mean(tab7["F míra průměrná"].values), decimals=3)))

########################################################################################################
clf8 = AdaBoostClassifier(n_estimators=1000, learning_rate=.1, random_state=0)
print("Počítám ADA model s {} estim a {} lr".format(clf8.n_estimators, clf8.learning_rate))

tab8 = PM.fast_CF_Boosted_Trees(model=clf8, Data=DATA, config=CONFIG_a, name="ADA_1000est_lr_0.1_{}".format(dodatek), location=valid_path_ADA, states=4)
print("F score average: {}".format(np.around(np.mean(tab8["F míra průměrná"].values), decimals=3)))

########################################################################################################
clf9 = AdaBoostClassifier(n_estimators=1000, learning_rate=.01, random_state=0)
print("Počítám ADA model s {} estim a {} lr".format(clf9.n_estimators, clf9.learning_rate))

tab9 = PM.fast_CF_Boosted_Trees(model=clf9, Data=DATA, config=CONFIG_a, name="ADA_1000est_lr_0.01_{}".format(dodatek), location=valid_path_ADA, states=4)
print("F score average: {}".format(np.around(np.mean(tab9["F míra průměrná"].values), decimals=3)))
########################################################################################################

print("Vše je hotovo!!!!!!")



