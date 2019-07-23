import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from time import time
from copy import copy
import pickle
import os
import sys
sys.path.append("../modules")

import Datasets as dat
import Feature_Engineering as FE
import Scoring
import Pipeline_Methods as PM

import hmmlearn
from hmmlearn.hmm import GaussianHMM

print("hmmlearn version:", hmmlearn.__version__)

path = os.getcwd()
path = path.replace(path.split('/')[-1],'')
valid_path_HMM = path + 'Tabulky a výsledky/Hidden Markov Model 2/'
# valid_path_SVC_test = path + 'Tabulky a výsledky/SVC/FirstDataset/'


def CF_HMM_modif(model, Data, configg, transmat, startprob, name, location, states):

    score_tab = Scoring.ScoringTable(location=location, name=name, n_states=states)

    df = PM.CreateDataFrame(Data=Data, config=configg)
    KFold = FE.KFold(Data.shape[2])

    history = {"TM":[], "Mean":[], "Cov":[]}

    start = time()
    for cross in range(Data.shape[0]):
        clf_un = copy(model)
        X_train, y_train, X_test, y_test = KFold.fit_transform(x=df, kFoldIndex=cross)

        lengths = np.copy(Data.shape[2])
        lengths = np.delete(lengths, cross)

        clf = PM.train_HMM_modif(model=clf_un, train=X_train, transmat=transmat, startprob=startprob, lengths=lengths, labels=y_train, n_states=states)

        pred = clf.predict(X_test)

        score = Scoring.score(states=pred, results=np.array(y_test), unsupervised=False, pocet_stavu=states)

        #print("score", score)
        score_tab.add(scores=score)
        print("{} section done. Time elapsed from start {}".format(cross+1, np.around(time()-start, decimals=2)))
        history["Mean"].append(clf.means_)
        history["Cov"].append(clf.covars_)
        history["TM"].append(clf.transmat_)


    score_tab.save_table()
    info = copy(configg)

    info["trasition_matrix"] = history["TM"]
    info["means"] = history["Mean"]
    info["covariance_matrix"] = history["Cov"]

    with open(location + name + '_config.pickle' , 'wb') as f:
        pickle.dump(info, f)

    return score_tab.return_table()


DATA = dat.load_dataset("first_dataset_mod")

CONFIG = {'H_alpha': True,
          '1.d SGF': True,
          '2.d SGF': True,
          'MM': [4, 6, 8, 10, 12, 14, 16],
          'EMM': [4, 6, 8, 10, 12, 14, 16],
          'MV': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}


N_STATES = 4

Transmatrix = np.array([[1/3, 1/3, 1/3, 0],
						[1/2, 1/2, 0, 0],
						[0, 0, 1/2, 1/2],
						[1/3, 1/3, 0, 1/3]])

Startprob = np.array([0,1,0,0])

MODEL = GaussianHMM(n_components=N_STATES, covariance_type="full", algorithm="viterbi", init_params='', params='mtc')

print("Začínáme!!")

tab = CF_HMM_modif(
                    model=MODEL,
                    Data=DATA,
                    configg=CONFIG,
                    transmat=Transmatrix,
                    startprob=Startprob,
                    name="hmm_all_feat",
                    location=valid_path_HMM,
                    states=N_STATES
)


print(tab)






