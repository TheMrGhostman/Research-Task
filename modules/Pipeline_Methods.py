# Pipeline_Methods
# Autor: Matěj Zorek
# Modul slouží k zrychlení tréninku a testování modelů

"""
TO DO:
	1) fix modules calling							DONE 13.4.2019
	2) fix for more states (or remake)				DONE 14.4.2019
	3) fix compatability for ScoringTable 			DONE 14.3.2019
	4) add train_batches
	5) fix compatability with GridSearch
	6) add save model method
	7) remake Cross-Fold method 					DONE 14.4.2019
	8) add function, for fast CF
		- vypočítám příznaky pro každý signal a 
		uložím je zvlášť v DataFramu				DONE 27.4.2019

"""

import numpy as np
import pandas as pd
import progressbar
import pickle
import itertools as it
from math import factorial
from copy import copy
from time import time
import Feature_Engineering as fe 
import Scoring 
import Datasets as d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid



def train_HMM_modif(model, train, transmat, lengths, labels, n_states = 3):
	"""
	transmat pro 3 stavy, kde 0 = h-mod, 1 = l-mod, 2 = elm, tzn. n_states=3
	model.transmat_ = (1/n_states)*np.ones((n_states, n_states))
	model.transmat_[1,:] = [1/2, 1/2, 0] <- zakazuju přechod z l-modu do elmu

	"""
	warnings.filterwarnings('ignore')

	model.init_params = ''
	model.startprob_ = np.array([0, 1, 0])
	model.means_, model.covars_ = fe.Preprocessing(train, n_states,
												   np.shape(train)[1],
												   labels)
	model.transmat_ = transmat
	model.fit(train, lengths)
	return model


def train_and_predict(model, train, test, labels, unsupervised):
	"""
	Až tuhle fci nahradím v ostatních funkcích, tak jí smažu.
	Dříve jsem jí používal pro šponavou práci s hmm. Nyní je nahrazen train_HMM_modif
	"""
	if unsupervised:
		model.fit(train)
	else:
		model.fit(train, labels)
	return model.predict(test)


def CF_Boosted_Trees(model, Data, configg, name, location, states=3): #train_data, test_data
	"""
	Input:
			train_data  ... list-of-lists 

	"""
	train_data, test_data = d.PrepareCrossFold(Data.H_alpha)
	train_labels, test_labels = d.PrepareCrossFold(Data.labels)

	
	K = len(train_data)
	score_tab = Scoring.ScoringTable(location=location, name=name, n_states=states)

	feat = fe.Features(config=configg, normalize=True)

	start = time()
	for cross in range(K):
		clf = copy(model)

		train_matrix = feat.fit_transform(Data = train_data[cross])
		test_matrix = feat.fit_transform_batch(Data=test_data[cross])
		target = d.merge_labels(train_labels[cross])

		#print(np.shape(train_matrix), np.shape(target))
		#print(np.shape(test_matrix), np.shape(test_labels[cross]))

		pred = train_and_predict(model=clf, train=train_matrix, test=test_matrix, 
								 labels= target, unsupervised=False)

		score = Scoring.score(states=pred, results=test_labels[cross], unsupervised=False, pocet_stavu=states)

		#print("score", score)

		score_tab.add(scores=score)
		print("{} section done. Time taken from start {}".format(cross+1, time()-start))

	score_tab.save_table()
	configg["n_estimators"] = model.n_estimators

	with open(location + name + '_config.pickle' , 'wb') as f:
		pickle.dump(configg, f)
	return score_tab.return_table()


def fast_CF_Boosted_Trees(model, Data, configg, name, location, states=3):
	"""
	Input:  Data  			... musí být Bunch z load_dataset()
	Output: score_tab		... hotová tabulka pd.DataFrame
	"""
	score_tab = Scoring.ScoringTable(location=location, name=name, n_states=states)

	df = CreateDataFrame(Data=Data, config=configg)
	KFold = fe.KFold(Data.shape[2])

	start = time()
	for cross in range(Data.shape[0]):
		clf = copy(model)
		X_train, y_train, X_test, y_test = KFold.fit_transform(x=df, kFoldIndex=cross)
		pred = train_and_predict(model=clf, train=X_train, test=X_test, 
								 labels= y_train, unsupervised=False)
		score = Scoring.score(states=pred, results=np.array(y_test), unsupervised=False, pocet_stavu=states)
		#print("score", score)
		score_tab.add(scores=score)
		print("{} section done. Time elapsed from start {}".fromat(cross+1, np.around(time()-start, decimals=2)))

	score_tab.save_table()
	info = copy(configg)
	info["n_estimators"] = model.n_estimators
	info["learning_rate"] = model.learning_rate

	with open(location + name + '_config.pickle' , 'wb') as f:
		pickle.dump(info, f)
	
	return score_tab.return_table()


def GridSearch(estimator, params, Data, config, name, location, states=3):

	df = CreateDataFrame(Data=Data, config=config)
	KFold = fe.KFold(Data.shape[2])

	start = time()
	GRID = ParameterGrid(params)
	combinations = len(list(GRID))
	print("Number of combinations {}".format(combinations))
	bar = progressbar.ProgressBar(maxval=combinations*10,
								  widgets=[progressbar.Bar('#', '[', ']'),
										   ' ', progressbar.Percentage()])
	combo=0
	bar.start()
	for i, g in enumerate(GRID):
		score_tab = Scoring.ScoringTable(location=location, name=name+str(g), n_states=states)
		for cross in range(Data.shape[0]):
			clf = copy(estimator)
			X_train, y_train, X_test, y_test = KFold.fit_transform(x=df, kFoldIndex=cross)
			clf.set_params(**g)
			pred = train_and_predict(model=clf, train=X_train, test=X_test, labels=y_train, unsupervised=False)
			score = Scoring.score(states=pred, results=np.array(y_test), unsupervised=False, pocet_stavu=states)
			score_tab.add(scores=score)
			combo +=1
			bar.update(combo)
		score_tab.save_table()
		del score_tab
		info = copy(config)
		info["params"] = g
		with open(location + name + str(i) + '_config.pickle' , 'wb') as f:
			pickle.dump(info, f)
			
	bar.finish()
	print('Celý proces trval: {} vteřin'.format(np.around(time()-start, decimals=0)))
	print('Hotovo!!')
	return 


def TreeBasedFeatureSelection(model, Data, config, name, location):
	"""
	Fce počítá feature_importances ze "stromových" modelů a vrací je v tabulce

	Input:  Data  			... musí být Bunch z load_dataset()
	Output: dg				... pd.DataFrame s feature importances
	"""
	df = CreateDataFrame(Data=Data, config=config)
	KFold = fe.KFold(Data.shape[2])
	FI = []
	start = time()
	for cross in range(Data.shape[0]):
		clf = copy(model)
		X_train, y_train, _, _ = KFold.fit_transform(x=df, kFoldIndex=cross)

		clf.fit(X_train, y_train)

		FI.append(np.round(clf.feature_importances_, decimals=4))
		print("{} section done. Time elapsed from start {}".fromat(cross+1, np.around(time()-start, decimals=2)))

	dg = pd.DataFrame(data=np.array(FI), columns=df.columns[:-1])
	dg = dg.transpose()
	if '.csv' not in name:
		name = name + '.csv'
	dg.to_csv(location + name)
	return dg


def CreateDataFrame(Data, config):
	"""
	Input:  Data 		 ... formát z load_dataset (Bunch)
			congigg      ... konfigurace

	Output: df           ... dataframe se všemi příznaky

	"""
	feat = fe.Features(config=config, normalize=True)
	X = feat.fit_transform(Data=Data.H_alpha)
	lab = d.merge_labels(labels=Data.labels)

	X = np.hstack((X, lab.reshape(lab.shape[0],1)))
	nm = feat.get_names(labels=True)

	df = pd.DataFrame(data = X, columns = nm)
	return df


def PrincipalComponentAnalysis(model, train_data, test_data, n_comp, configg, model_nm, dataset_nm):

	model_name = f"{model_nm}_{dataset_nm}_dataset.csv"
	table = ScoringcoringTable(name = model_name, location = valid_path)

	for cross in range(10):
		start = time()
		train_matrix = fe.prepare_features(data=train_data[cross], config=configg)
		test_matrix = fe.prepare_features(data=test_data[cross], config=configg)
		labels = d.merge_labels(d.get_layer(train_data[cross],2))

		clf = copy(model)

		scale = StandardScaler()
		train_matrix = scale.fit_transform(X=train_matrix, y=labels)
		test_matrix = scale.transform(X=test_matrix)

		pca = PCA(n_components=n_comp)
		train_matrix = pca.fit_transform(X=train_matrix)
		test_matrix = pca.transform(X=test_matrix)

		states = train_and_predict(clf,
								   train_matrix, test_matrix, [], labels,
								   unsupervised=False, HMMmodified=False)

		[a, m, f, f_a, p, r] = Scoring.score(states, test_data[cross][0][2], unsupervised=False)

		kombinace = "pca"
		params = "pca"

		table.add(scores = [a, m, f, f_a, p, r], 
				  n_estim = clf.n_estimators, 
				  configg={"Komb": kombinace, "Param": params})

		print(cross + 1, '. -> ', round(time()-start), "seconds")

	table.save_table()

	return table.return_table()

