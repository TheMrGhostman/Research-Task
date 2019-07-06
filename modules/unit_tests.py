import numpy as np
import pandas as pd
import os
from bunch import Bunch
from time import time, sleep
import Feature_Engineering
import Datasets
import Scoring
import Pipeline_Methods as PM
from sklearn.ensemble import AdaBoostClassifier
import pickle


def load_data():
	Data = Datasets.load_dataset("first_dataset")
	X = Data.data
	x = Data.H_alpha
	y = Data.labels
	return X,x,y


#X,x,y = load_data()

CONFIG = {"H_alpha": True,
		  "1.d SGF": True,
		  "2.d SGF": True,
		  "MM": [4,6,8,10,12,14,16],
		  "EMM": [4,6,8,10,12,14,16],
		  "MV": [5,6,7,8,9,10,11,12,13,14,15,16]}

"""
	prepare_features vs Feature.fit_trainsform (class)
"""
def test_prepare_features(X,x,y):
	st = time()
	PF = Feature_Engineering.prepare_features(X, config=CONFIG)
	en = time()
	print("prepare_features ... time = ", en-st)


	feat = Feature_Engineering.Features(config=CONFIG, normalize=True)
	st = time()
	FC = feat.fit_transform(x)
	en = time()
	print("fit_transform ... time = ", en-st)


	print("Shodují se výstupy prepare_features a Features.fit_transform? ", np.allclose(PF,FC))


# test_prepare_features(X=X,x=x,y=y)

"""
Test PrepareCrossFold jen s Bunch.H_alpha
"""
def test_PrepareCrossFold(Data, vypis= False):
	train, test = Datasets.PrepareCrossFold(Data.H_alpha)
	tmp1, tmp2 = Datasets.PrepareCrossFold(Data.data)

	shodujiSe = True

	for m,n in enumerate(train):
		for i,j in enumerate(n):
			YN = np.allclose(j, tmp1[m][i][1])
			if YN == False:
				shodujiSe = False
			if vypis:
				print(f'{m} Souhlasí {i+1} signál? ', YN)
	print("Shodují se všechny? ", shodujiSe)


# test_PrepareCrossFold(Data=Data)


def test_CF_BT(data):
	model = AdaBoostClassifier(n_estimators= 10)
	tab = PM.CF_Boosted_Trees(model = model, 
								Data=data, 
								configg=CONFIG , 
								name = '/unit_test', 
								location = os.getcwd(), 
								states=3)
	print(tab.head())

	with open(os.getcwd()+'/unit_test'+ '_config.pickle' , 'rb') as f:
		print(pickle.load(f))


#test_CF_BT(Data)

"""
tmp1, tmp2 = Datasets.PrepareCrossFold(Data.data)
train, test = Datasets.PrepareCrossFold(Data.H_alpha)

print(tmp2[0][2])
print(test)
"""


def test_progressbar(max_val, sleep_time=2):
	bar = Scoring.Bar(max_val)
	bar.start()
	for i in range(max_val):
		bar.update(i+1)
		sleep(sleep_time)
	bar.finish()
	print("all is good")


test_progressbar(60)
