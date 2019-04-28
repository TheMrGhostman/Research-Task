import Classification as CL
import numpy as np 
import pandas as pd 
import datasets as d 
import os 
import progressbar
from itertools import chain
from time import time
from copy import copy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
import sys
#import warnings

#setting path for saving of validations tabs and for import os
path = os.getcwd()
path = path.replace(path.split('/')[-1],'') # pro windows je třeba "\\" u linuxu stačí "/"
valid_path = path + "PCA/"


CONFIG = {"H_alpha": True, 
		  "1.d SGF": True, 
		  "2.d SGF": True, 
		  "MM": [4,6,8,10,12,14,16], 
		  "EMM": [4,6,8,10,12,14,16], 
		  "MV": [5,6,7,8,9,10,11,12,13,14,15,16]}


def PrincipalComponentAnalysis(model, train_data, test_data, n_comp, configg, model_nm, dataset_nm):

	model_name = f"{model_nm}_{dataset_nm}_dataset.csv"
	table = CL.ScoringTable(name = model_name, location = valid_path)

	for cross in range(10):
		start = time()
		train_matrix = CL.prepare_features(data=train_data[cross], config=configg)
		test_matrix = CL.prepare_features(data=test_data[cross], config=configg)
		labels = d.merge_labels(d.get_layer(train_data[cross],2))

		clf = copy(model)

		scale = StandardScaler()
		train_matrix = scale.fit_transform(X=train_matrix, y=labels)
		test_matrix = scale.transform(X=test_matrix)

		pca = PCA(n_components=n_comp)
		train_matrix = pca.fit_transform(X=train_matrix)
		test_matrix = pca.transform(X=test_matrix)

		states = CL.train_and_predict(clf,
		                              train_matrix, test_matrix, [], labels,
		                              unsupervised=False, HMMmodified=False)

		[a, m, f, f_a, p, r] = CL.score(states, test_data[cross][0][2], unsupervised=False)

		kombinace = "pca"
		params = "pca"

		table.add(scores = [a, m, f, f_a, p, r], 
				  n_estim = clf.n_estimators, 
				  configg={"Komb": kombinace, "Param": params})

		print(cross + 1, '. -> ', round(time()-start), "seconds")

	table.save_table()

	return table.return_table()



if sys.argv[1] == "first":
    DATA = d.load_dataset(name="first_dataset")
    #CONFIG=CONFIG1
elif sys.argv[1] == "second":
    DATA = d.load_dataset(name="second_dataset")
    #CONFIG=CONFIG2
else:
    raise ValueError("Pouze dva datasety: first nebo second!!")

TRAIN, TEST = d.PrepareCrossFold(DATA.data)

LR = float(sys.argv[3])
NESTIM = int(sys.argv[2])
NCOMP = int(sys.argv[4])

tebulky_pca = PrincipalComponentAnalysis(model = AdaBoostClassifier(n_estimators = NESTIM, learning_rate = LR, random_state = 0), 
										 train_data = TRAIN, 
										 test_data = TEST, 
										 n_comp=NCOMP, 
										 configg=CONFIG, 
										 model_nm = f"ADA_{NESTIM}_lr_{LR}_PCAcomp={NCOMP}", 
										 dataset_nm=f"{sys.argv[1]}")

"""
AdaBoostClassifier
GradientBoostingClassifier
"""
print("Hotovo!!!")

