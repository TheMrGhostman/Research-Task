# Scoring
# Autor: Matěj Zorek
# Modul slouží k výpočtům metrik, ohodnocení kvality modelu a zpracování výsledků

"""
 TO DO:
	1) fix functions fro more states.  								DONE 13.4.2019
	2) remake "add" function or add another function for 
		preparing inputs.

"""


import numpy as np
import pandas as pd
from math import factorial, isnan
from copy import copy
from sympy.utilities.iterables import multiset_permutations
import warnings
import progressbar


class Bar:
	def __init__(self, max_val):
		self.bar = progressbar.ProgressBar(
											maxval=max_val, widgets=[progressbar.Bar('#', '[', ']'),
											' ', progressbar.Percentage()]
		)

	def start(self):
		self.bar.start()

	def update(self, iterace):
		self.bar.update(iterace)

	def finish(self):
		self.bar.finish()


def srovnej(res, data, pocet_stavu=3):
	"""
	Permutační funkce vracející nejlepší přerovnaný výsledek podle porovnání se
		skutečnými výsledky (labely)
	! V současné formě je schopna přerovnat až 3 skupiny(stavy). NÉ VÍCE!

	Input: res         .. vektor výsledků, k němuž chceme najít nejlepší
						  přerovnání dat (np.array() "1D")
		   data        .. vektor dat pro přerovnání (np.array "1D")
		   pocet_stavu .. počet stavů (skupin) (int z {2,3})

	Output: [suma správně, správně přerovnaná data]
					   .. přerovnaná data (np.array() "1D")
	"""
	# já vlastně přehazju data tak aby byla co největší schoda s res (skutečné výsledky)
	# když za data zadám stavy dostanu pak vektor stavů přerovnaný podle nám známého řešení(res)
	# to znamená že se bych je pak mohl přiradit správně do tříd, pokud bude známo řešení dopředu

	if pocet_stavu == 2:
		[temp0, temp1] = np.unique(data)

		reverse = np.copy(data)
		reverse[reverse == temp0] = -1
		reverse[reverse == temp1] = temp0
		reverse[reverse == -1] = temp1

		right_sorted = [sum(res == data), sum(res == reverse)]
		vector = np.vstack((data, reverse))
		return([max(right_sorted), vector[np.argmax(right_sorted), :]])
	else:
		if len(np.unique(data)) == 2:
			#srovnej(res, data, 2)
			[temp0, temp1] = np.unique(data)
			temp2 = copy(temp1)
		else:
			[temp0, temp1, temp2] = np.unique(data)
		perm = np.array([temp0, temp1, temp2])
		right_sorted = []
		vector = np.copy(data)
		for i in range(factorial(pocet_stavu) - 1):
			vector = np.vstack((vector, np.copy(data)))
		j = 0
		for p in multiset_permutations(perm):
			vector[j][vector[j] == temp0] = -1
			vector[j][vector[j] == temp1] = -2
			vector[j][vector[j] == temp2] = -3
			vector[j][vector[j] == -1] = p[0]
			vector[j][vector[j] == -2] = p[1]
			vector[j][vector[j] == -3] = p[2]
			right_sorted.append(sum(vector[j] == res))
			j += 1
		return([max(right_sorted), vector[np.argmax(right_sorted), :]])


def Accuracy(výsledek, stavy, pocet_stavu, srovnat=True, CM=[False]):
	"""
	Funkce počítá přesnost klasifikace metody (modelu)
		- slouží taky jako rozhodovací parametr pro funkci srovnej()

	Input: výsledek    ... výstup predikce modelu (np.array "1D")
		   stavy       ... skutečné labely resp. správné řešení (np.array() "1D")
		   pocet_stavu ... integer udávající počet stavů (int)
		   srovnat     ... parametr určuje, zda je třeba nejdříve přerovnat
						   výsledky podel skutečných stavů (bool)
		   CM          ... je nepovinný parametr typu list, kde na první pozici
						   je True nebo False, a na druhé je již vypočítaná
						   Confusion Matrix. Parametr je nepovinný, ale zabraňuje
						   zbytečnému přepočítání CM a zrychluje vyhodnocení
						   ([bool, np.matrix()])

	Output: output ... list skládající se z přesnosti modelu a počtu chyb
					   ([float64, int])
	"""
	if CM[0]:
		spravne = sum(CM[1].diagonal())
		return [spravne/len(stavy), int(len(stavy)-spravne)]
	else:
		if srovnat:
			if len(výsledek) != len(stavy):
				print("stavy a výsledky nesouhlasí dimenze")
				return
				#print(Confusion_Matrix(výsledek, stavy, pocet_stavu))
			if pocet_stavu <= 2:
					#předpokládám že pří dvou stvech nebudu mít víc chyb než správných klasifikací
				součet = max(sum(výsledek == stavy), sum(výsledek != stavy))
				return [součet/len(stavy), len(stavy) - součet]
			else:
				přesnost = srovnej(výsledek, stavy)[0]
				return [přesnost / len(výsledek), int(len(výsledek) - přesnost)]
		else:
			sou = sum(výsledek == stavy)
			return [sou / len(stavy), int(len(stavy) - sou)]


def Confusion_Matrix(výsledek, stavy, pocet_stavu, srovnat = True):
	"""
	Funkce vytváří matici záměn tzn. Confusion matrix, která je potřebná pro
		výpočty všech vyhodnocovacích metrik používaných při experimentech

	Input: výsledek    ... výstup predikce modelu (np.array "1D")
		   stavy       ... skutečné labely resp. správné řešení (np.array() "1D")
		   pocet_stavu ... integer udávající počet stavů (int)
		   srovnat     ... parametr určuje, zda je třeba nejdříve přerovnat
						   výsledky podel skutečných stavů (bool)

	Output: tabulka    ... vytořená matice záměn (np.matrix())
	"""
	if srovnat == True:
		srovnaný = srovnej(výsledek, stavy, pocet_stavu)[1]
		#print(srovnaný)
		#print(výsledek)
	else:
		srovnaný = stavy
	tabulka = np.zeros((pocet_stavu, pocet_stavu), dtype='int64')
	for i in range(pocet_stavu):
		for j in range(pocet_stavu):
			tabulka[i, j] = sum(výsledek[k] == i and srovnaný[k] == j for k in range(len(výsledek)))
	return tabulka


def F_Measure(výsledek, stavy, pocet_stavu, srovnat=True, CM=[False]):
	"""
	Funkce počítá F - míru, pomocí které hodnonotíme kvalitu modelu

	Input: výsledek    ... výstup predikce modelu (np.array "1D")
		   stavy       ... skutečné labely resp. správné řešení (np.array() "1D")
		   pocet_stavu ... integer udávající počet stavů (int)
		   srovnat     ... parametr určuje, zda je třeba nejdříve přerovnat
						   výsledky podel skutečných stavů (bool)
		   CM          ... je nepovinný parametr typu list, kde na první pozici
						   je True nebo False, a na druhé je již vypočítaná
						   Confusion Matrix. Parametr je nepovinný, ale zabraňuje
						   zbytečnému přepočítání CM a zrychluje vyhodnocení
						   ([bool, np.matrix()])

	Output: output     ... list skládající se z F-míry pro jedotlivé stavy a
						   hodnoty průměrné F-míry ([np.array(), float])
	"""
	if CM[0]:
		tabulka = CM[1]
	else:
		tabulka = Confusion_Matrix(výsledek, stavy, pocet_stavu, srovnat)

	FM = np.zeros(pocet_stavu)
	for k in range(pocet_stavu):
		FM[k] = 2 * tabulka[k, k] / (sum(tabulka[k, :]) + sum(tabulka[:, k]))
	# vracím [FM vektor, FM macro]
	# FM je F1_score('none') a suma je F1_score("macro") což je průměr F1 z všech tříd
	#print(tabulka)
	return [FM, sum(FM) / pocet_stavu]


def Precision_n_Recall(výsledek, stavy, pocet_stavu, srovnat=True, CM=[False]):
	"""
	Funkce počítá Precision a Recall pro jedotlivé stavy

	Input: výsledek    ... výstup predikce modelu (np.array "1D")
		   stavy       ... skutečné labely resp. správné řešení (np.array() "1D")
		   pocet_stavu ... integer udávající počet stavů (int)
		   srovnat     ... parametr určuje, zda je třeba nejdříve přerovnat
						   výsledky podel skutečných stavů (bool)
		   CM          ... je nepovinný parametr typu list, kde na první pozici
						   je True nebo False, a na druhé je již vypočítaná
						   Confusion Matrix. Parametr je nepovinný, ale zabraňuje
						   zbytečnému přepočítání CM a zrychluje vyhodnocení
						   ([bool, np.matrix()])

	Output: output     ... list skládající se z vektrou hodnot precisionů 
						   jednotlivých stavů a vektoru hodnot recallů 
						   jednotlivých stavů ([np.array(), np.array()])
	"""
	if CM[0]:
		tabulka = CM[1]
	else:
		tabulka = Confusion_Matrix(výsledek, stavy, pocet_stavu, srovnat)
	precision = np.zeros(pocet_stavu)
	recall = np.zeros(pocet_stavu)
	for k in range(pocet_stavu):
		precision[k] = tabulka[k, k] / sum(tabulka[:, k])
		recall[k] = tabulka[k, k] / sum(tabulka[k, :])
		if isnan(precision[k]):
			precision[k] = 0
			#print("Precision was NaN ")
		if isnan(recall[k]):
			recall[k] = 0
			#print("Recall was NaN")
	#print(tabulka)
	return [precision, recall]


def score(states, results, unsupervised, pocet_stavu=3):
	Conf_Mat = Confusion_Matrix(results, states, pocet_stavu, unsupervised)
	[Acc, M] = Accuracy(results, states, pocet_stavu, unsupervised, [True, Conf_Mat])
	[F, F_a] = F_Measure(results, states, pocet_stavu, unsupervised, [True, Conf_Mat])
	[P, R] = Precision_n_Recall(results, states, pocet_stavu, unsupervised, [True, Conf_Mat])
	return [Acc, M, F, [F_a], P, R]


class ScoringTable:
	"""	
	Třída ScoringTable byla vytvořena pro zjednodušení vytváření tabulek pro vyhodnocení
	kvality modelů a uchování výsledků

	"""
	warnings.filterwarnings('ignore')

	def __init__(self, name, location, n_states=3):
		self.location = location
		self.name = name
		self.dataframe = None
		self.n_states = n_states
		self.columns = self._columns()


	def _columns(self):
		cols = ['Accuracy', 'Chyby']
		for j in ['F míra', 'Precision', 'Recall']:
			for i in range(self.n_states):
				cols.append('{} stavu {}'.format(j,i))
			if j == 'F míra':
				cols.append('{} průměrná'.format(j))
		return cols


	def list_to_numpy(self, listt):
		if len(listt)!=6:
			raise ValueError("Chybý nějaká z metrik!")

		output = np.array([listt[0], listt[1]])
		tmp = [i for j in listt[2:] for i in j]
		return np.hstack((output, tmp))


	def add(self, scores, n_estim=None, configg=None):
		"""
		configg ve formě {"Komb":[], "Param": []}

		"""

		score_row = self.list_to_numpy(listt=scores)

		if configg:
			if not 'Délky úseku' in self.columns:
				self.columns.insert(0, 'Délky úseku')
				self.columns.insert(0, 'Kombinace rysů')
			score_row = np.hstack((np.array([configg["Param"]]), score_row))
			score_row = np.hstack((np.array([configg["Komb"]]), score_row))
			
		if n_estim:
			if not 'n_estim' in self.columns:
				self.columns.insert(0, 'n_estim')
			score_row = np.hstack((np.array([n_estim]), score_row))
		
		#print(isinstance(self.dataframe, np.ndarray))
		if not isinstance(self.dataframe, np.ndarray):#self.dataframe == None:
			self.dataframe = score_row
		else:
			self.dataframe = np.vstack((self.dataframe, score_row))


	def return_table(self):
		return pd.DataFrame(data=self.dataframe, columns=self.columns)


	def save_table(self, location=None):
		df = self.return_table()
		if ".csv" not in self.name:
			self.name = self.name+'.csv'
		if self.location == None:
			self.location = location
		df.to_csv(self.location + self.name , index=False)


