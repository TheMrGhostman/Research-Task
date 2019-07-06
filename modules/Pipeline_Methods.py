# Pipeline_Methods
# Autor: Matěj Zorek
# Modul slouží k zrychlení tréninku a testování modelů

"""
TO DO:
	1) fix modules calling							DONE 13.4.2019
	2) fix for more states (or remake)				DONE 14.4.2019
	3) fix compatibility for ScoringTable 			DONE 14.3.2019
	4) add train_batches
	5) fix compatibility with GridSearch
	6) add save model method
	7) remake Cross-Fold method 					DONE 14.4.2019
	8) add function, for fast CF
		- vypočítám příznaky pro každý signal a 
		uložím je zvlášť v DataFrame				DONE 27.4.2019
	9) add cross-fold method for HMM_modif 			DONE 12.6.2019

"""

import numpy as np
import pandas as pd
import progressbar
import pickle
import warnings
import itertools as it
from math import factorial
from copy import copy
from time import time
import Feature_Engineering as FE
import Scoring
import Datasets as dat
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.feature_selection import SelectKBest


def train_HMM_modif(model, train, transmat, start_prob, lengths, labels, n_states=3):
	"""
	transmat pro 3 stavy, kde 0 = h-mod, 1 = l-mod, 2 = elm, tzn. n_states=3
	model.transmat_ = (1/n_states)*np.ones((n_states, n_states))
	model.transmat_[1,:] = [1/2, 1/2, 0] <- zakazuji přechod z l-modu do elmu

	"""
	warnings.filterwarnings('ignore')

	model.init_params = ''
	model.startprob_ = start_prob # np.array([0, 1, 0])
	train = np.matrix(train)
	labels = np.array(labels)
	model.means_, model.covars_ = FE.Preprocessing(data=train, pocet_stavu=n_states,
												   pocet_feature=np.shape(train)[1], labels=labels)
	model.transmat_ = transmat

	if np.sum(lengths) != train.shape[0]:
		print("Něco je špatně!! Délky neodpovídají počtu pozorování!!")

	model.fit(X=train, lengths=lengths)

	return model


def CF_HMM_modif(model, Data, config, transmat, startprob, name, location, states):
	"""
	Funkce pro provedení Cross-fold validace Hidden Markov Modelu (modifikovaného).

	Input: model 			... GaussianHMM s již definovanými parametry
			Data 			... list-of-lists (Datasets.load_dataset format)
			config 			... konfigurace příznaků
			trans_mat 		... předpočítává matice přechodů, resp. její inicializace
			start_prob 		... pravděpodobnost prvního stavu
			states 			... počet stavů

	"""
	score_tab = Scoring.ScoringTable(location=location, name=name, n_states=states)

	feature = FE.Features(config=config, normalize=True)
	df = feature.CreateDataFrame(Data=Data)
	# df = CreateDataFrame(Data=Data, config=config)
	KFold = FE.KFold(Data.shape[2])

	history = {"TM":[], "Mean":[], "Cov":[]}

	start = time()
	for cross in range(Data.shape[0]):
		clf_un = copy(model)
		X_train, y_train, X_test, y_test = KFold.fit_transform(x=df, kFoldIndex=cross)

		lengths = np.copy(Data.shape[2])
		lengths = np.delete(lengths, cross)

		clf = train_HMM_modif(model=clf_un, train=X_train, transmat=transmat, start_prob=startprob,
							  lengths=lengths, labels=y_train, n_states=states)

		y_pred = clf.predict(X_test)

		score = Scoring.score(states=y_pred, results=np.array(y_test), unsupervised=False, pocet_stavu=states)

		# print("score", score)
		score_tab.add(scores=score)
		print("{} section done. Time elapsed from start {}".format(cross+1, np.around(time()-start, decimals=2)))
		history["Mean"].append(clf.means_)
		history["Cov"].append(clf.covars_)
		history["TM"].append(clf.transmat_)


	score_tab.save_table()
	info = copy(config)

	info["trasition_matrix"] = history["TM"]
	info["means"] = history["Mean"]
	info["covariance_matrix"] = history["Cov"]

	with open(location + name + '_config.pickle', 'wb') as f:
		pickle.dump(info, f)

	return score_tab.return_table()


def train_and_predict(model, train, test, labels, unsupervised):
	"""
	Až tuhle fci nahradím v ostatních funkcích, tak jí smažu.
	Dříve jsem jí používal pro špinavou práci s hmm. Nyní je nahrazen train_HMM_modif
	"""
	if unsupervised:
		model.fit(train)
	else:
		model.fit(train, labels)
	return model.predict(test)


def CF_Boosted_Trees(model, Data, config, name, location, states=3):  # train_data, test_data
	"""
	Input:
			train_data  ... list-of-lists 

	"""
	train_data, test_data = dat.PrepareCrossFold(Data.H_alpha)
	train_labels, test_labels = dat.PrepareCrossFold(Data.labels)

	K = len(train_data)
	score_tab = Scoring.ScoringTable(location=location, name=name, n_states=states)

	feat = FE.Features(config=config, normalize=True)

	start = time()
	for cross in range(K):
		clf = copy(model)

		train_matrix = feat.fit_transform(Data = train_data[cross])
		test_matrix = feat.fit_transform_batch(Data=test_data[cross])
		target = dat.merge_labels(train_labels[cross])

		# print(np.shape(train_matrix), np.shape(target))
		# print(np.shape(test_matrix), np.shape(test_labels[cross]))

		pred = train_and_predict(model=clf, train=train_matrix, test=test_matrix, labels=target, unsupervised=False)

		score = Scoring.score(states=pred, results=test_labels[cross], unsupervised=False, pocet_stavu=states)

		# print("score", score)

		score_tab.add(scores=score)
		print("{} section done. Time taken from start {}".format(cross+1, time()-start))

	score_tab.save_table()
	config["n_estimators"] = model.n_estimators

	with open(location + name + '_config.pickle' , 'wb') as f:
		pickle.dump(config, f)
	return score_tab.return_table()


def fast_CF_Boosted_Trees(model, Data, config, name, location, states=3):
	"""
	Input:  Data  			... musí být Bunch z load_dataset()
	Output: score_tab		... hotová tabulka pd.DataFrame
	"""
	score_tab = Scoring.ScoringTable(location=location, name=name, n_states=states)

	feature = FE.Features(config=config, normalize=True)
	df = feature.CreateDataFrame(Data=Data)
	# df = CreateDataFrame(Data=Data, config=config)
	KFold = FE.KFold(Data.shape[2])

	start = time()
	for cross in range(Data.shape[0]):
		clf = copy(model)
		X_train, y_train, X_test, y_test = KFold.fit_transform(x=df, kFoldIndex=cross)
		y_pred = train_and_predict(model=clf, train=X_train, test=X_test, labels=y_train, unsupervised=False)
		score = Scoring.score(states=y_pred, results=np.array(y_test), unsupervised=False, pocet_stavu=states)
		# print("score", score)
		score_tab.add(scores=score)
		print("{} section done. Time elapsed from start {}".format(cross+1, np.around(time()-start, decimals=2)))

	score_tab.save_table()
	info = copy(config)
	info["n_estimators"] = model.n_estimators
	info["learning_rate"] = model.learning_rate

	with open(location + name + '_config.pickle', 'wb') as f:
		pickle.dump(info, f)

	return score_tab.return_table()


def GridSearch(estimator, params, Data, config, name, location, states=3):

	df = CreateDataFrame(Data=Data, config=config)
	KFold = FE.KFold(Data.shape[2])

	start = time()
	GRID = ParameterGrid(params)
	combinations = len(list(GRID))
	print("Number of combinations {}".format(combinations))
	bar = progressbar.ProgressBar(maxval=combinations*10, widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
	combo = 0
	bar.start()
	for i, g in enumerate(GRID):
		score_tab = Scoring.ScoringTable(location=location, name=name+str(g)+str(i), n_states=states)
		for cross in range(Data.shape[0]):
			clf = copy(estimator)
			X_train, y_train, X_test, y_test = KFold.fit_transform(x=df, kFoldIndex=cross)
			clf.set_params(**g)
			pred = train_and_predict(model=clf, train=X_train, test=X_test, labels=y_train, unsupervised=False)
			score = Scoring.score(states=pred, results=np.array(y_test), unsupervised=False, pocet_stavu=states)
			score_tab.add(scores=score)
			combo += 1
			bar.update(combo)
		score_tab.save_table()
		del score_tab
		info = copy(config)
		info["params"] = g
		with open(location + name + str(g) + str(i) + '_config.pickle', 'wb') as f:
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
	KFold = FE.KFold(Data.shape[2])
	FI = []
	start = time()
	for cross in range(Data.shape[0]):
		clf = copy(model)
		X_train, y_train, _, _ = KFold.fit_transform(x=df, kFoldIndex=cross)

		clf.fit(X_train, y_train)

		FI.append(np.round(clf.feature_importances_, decimals=4))
		print("{} section done. Time elapsed from start {}".format(cross+1, np.around(time()-start, decimals=2)))

	dg = pd.DataFrame(data=np.array(FI), columns=df.columns[:-1])
	dg = dg.transpose()
	if '.csv' not in name:
		name = name + '.csv'
	dg.to_csv(location + name)
	return dg


def FeatureScoring(data, config, method, printout=True):
	"""
	Funkce je určena k Cross-Fold ohodnocení příznaků s využitím SelectKBest


	Input:  Data 		 ... formát z load_dataset (Bunch)
			config      ... konfigurace
			method 		 ... metoda (mutual_info_classif, f_classif)

	Output: df           ... dataframe se všemi příznaky

	"""
	History = []

	feature = FE.Features(config=config, normalize=True)
	df = feature.CreateDataFrame(Data=data)
	# df = CreateDataFrame(Data=data, config=config)
	KFold = FE.KFold(data.shape[2])

	for cross in range(data.shape[0]):
		X_train, y_train, _,_ = KFold.fit_transform(x=df, kFoldIndex=cross)
		KB = SelectKBest(method, k="all")
		KB.fit(X_train, y_train)
		History.append(KB.scores_)
		if printout:
			print("Step {}/{} done.".format(cross+1, data.shape[0]))

	History = pd.DataFrame(data=np.matrix(History).T, index=list(df.columns)[:-1], columns=np.arange(data.shape[0])+1)
	History.loc[:, "average"] = History.values.mean(axis=1)

	return History


def CreateDataFrame(Data, config):
	"""
	Input:  Data 		 ... formát z load_dataset (Bunch)
			config      ... konfigurace

	Output: df           ... dataframe se všemi příznaky

	"""
	feat = FE.Features(config=config, normalize=True)
	X = feat.fit_transform(Data=Data.H_alpha)
	lab = dat.merge_labels(labels=Data.labels)

	X = np.hstack((X, lab.reshape(lab.shape[0],1)))
	nm = feat.get_names(labels=True)

	df = pd.DataFrame(data=X, columns=nm)
	return df


def PrincipalComponentAnalysis(model, Data, n_comp, config, name, location, states=3, model_type="Tree"):

	score_tab = Scoring.ScoringTable(name=name, location=location)
	feature = FE.Features(config=config, normalize=True)
	df = feature.CreateDataFrame(Data=Data)

	KFold = FE.KFold(Data.shape[2])

	start = time()
	for cross in range(Data.shape[0]):
		clf = copy(model)
		X_train, y_train, X_test, y_test = KFold.fit_transform(x=df, kFoldIndex=cross)

		scale = StandardScaler()
		X_train = scale.fit_transform(X=X_train, y=y_train)
		X_test = scale.transform(X=X_test)

		pca = PCA(n_components=n_comp)
		X_train = pca.fit_transform(X=X_train)
		X_test = pca.transform(X=X_test)

		y_pred = train_and_predict(model=clf, train=X_train, test=X_test, labels=y_train, unsupervised=False)

		score = Scoring.score(states=y_pred, results=np.array(y_test), unsupervised=False, pocet_stavu=states)

		score_tab.add(scores=score)
		print("{} section done. Time elapsed from start {}".format(cross + 1, np.around(time() - start, decimals=2)))

	score_tab.save_table()

	info = copy(config)
	info["pca component"] = n_comp
	if model_type == "Tree":
		info["n_estimators"] = model.n_estimators
		info["learning_rate"] = model.learning_rate
	if model_type == "SVM":
		info["C"] = model.C
		info["kernel"] = model.kernel
		if model.kernel != 'linear':
			info["gamma"] = model.gamma

	with open(location + name + '_config.pickle', 'wb') as f:
		pickle.dump(info, f)

	return score_tab.return_table()


def GridSearchPCA(estimator, params, Data, config, n_comp, name, location, states=3):
	df = CreateDataFrame(Data=Data, config=config)
	KFold = FE.KFold(Data.shape[2])

	start = time()
	GRID = ParameterGrid(params)
	combinations = len(list(GRID))
	print("Number of combinations {}".format(combinations))
	bar = Scoring.Bar(combinations*10)
	combo = 0
	bar.start()
	for i, g in enumerate(GRID):
		score_tab = Scoring.ScoringTable(location=location, name=name+str(g)+str(i), n_states=states)
		for cross in range(Data.shape[0]):
			clf = copy(estimator)
			X_train, y_train, X_test, y_test = KFold.fit_transform(x=df, kFoldIndex=cross)

			scale = StandardScaler()
			X_train = scale.fit_transform(X=X_train, y=y_train)
			X_test = scale.transform(X=X_test)

			pca = PCA(n_components=n_comp)
			X_train = pca.fit_transform(X=X_train)
			X_test = pca.transform(X=X_test)

			clf.set_params(**g)
			pred = train_and_predict(model=clf, train=X_train, test=X_test, labels=y_train, unsupervised=False)
			score = Scoring.score(states=pred, results=np.array(y_test), unsupervised=False, pocet_stavu=states)
			score_tab.add(scores=score)
			combo += 1
			bar.update(combo)

		score_tab.save_table()
		del score_tab
		info = copy(config)
		info["params"] = g
		with open(location + name + str(g) + str(i) + '_config.pickle', 'wb') as f:
			pickle.dump(info, f)

	bar.finish()
	print('Celý proces trval: {} vteřin'.format(np.around(time()-start, decimals=0)))
	print('Hotovo!!')
	return

