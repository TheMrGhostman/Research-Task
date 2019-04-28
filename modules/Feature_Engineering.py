# Feature_Engineering
# Autor: Matěj Zorek
# Modul slouží k výpočtům příznaků a obecně předzpracování datasetu

"""
TO DO:
	1) figure out how to train batches
			- malé množství přes fit_transform_batch	DONE 14.4.2019
			- větší množství se nejdříve vypočítá 
				buď hromadně nebo přes batche
				a uloží se do h.5 - do tréninku ho 
				budu načítat po částech					DONE 14.4.2019
	2) finish Feature class 							DONE 14.4.2019
	3) classmethod all_features 						DONE 13.4.2019
	4) fix inputs format in prepare_features
		- vytvořil jsem fit_transform, kde je to už
			opravené									DONE 14.4.2019	
	5) add changable parameters for SGF
		- přidání self.savgol_paramns umožňuje ve
			třídě měnit i další parametry				DONE 14.4.2019
	6) add function, which creates batches
		- vypočítám příznaky pro každý signal a 
			uložím je zvlášť v DataFramu (tzn KFold)	DONE 26.4.2019
	7) add KFold class 									
		- zrychlení operace s pozorováními				DONE 26.4.2019
"""


#import itertools as it
import numpy as np
import pandas as pd
from math import factorial
from copy import copy
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs
from numba import guvectorize
from bunch import Bunch

"""
	Fce pro výpočty příznaků
"""

@guvectorize(['void(float64[:], int64, float64[:])'], '(n),()->(n)')
def derivace(data, krok, deriv):
	"""
	Výpočet derivace pomocí centrální diference 2. řádu.

	Input: data .. vektor dat (np.array() "1D")
		   krok .. časový krok "h" (float64)

	Output: deriv .. vypočtená derivace pro všechny hodnoty z vektrou data
					   (np.array() "1D")
	"""
	data = np.array(data)
	kon = len(data)-1
	deriv[0] = (data[1]-data[0])/krok
	deriv[kon] = (data[kon]-data[kon-1])/krok
	for i in range(1, len(data)-1):
		deriv[i] = (data[i+1]-data[i-1])/(2*krok)


@guvectorize(['void(float64[:], int64, float64[:])'], '(n),()->(n)')
def exp_moving_mean(data, window, emm):
	"""
	Výpočet exponenciálně tlumeného klouzavého průměru (Exponential moving mean)

	Input: data   .. vektor dat (np.array() "1D")
		   window .. časový úsek, na kterém je počítán emm " (int)

	Output: emm .. vypočtený exponenciální klouzavý průměr pro všechny hodnoty
				   z vektrou data(np.array() "1D")
	"""
	gamma = 0.9**np.arange(window)
	gamm = 0.9**np.arange(window)[::-1]
	count = 0
	dolni_index = 0
	for i in range(window):
		count += 1
		emm[i] = (sum(data[dolni_index: i + 1]*gamma[dolni_index: i + 1][::-1]))*(1/count)
	for i in range(window, len(data)):
		dolni_index = i + 1 - window
		emm[i] = (sum(data[dolni_index: i + 1]*gamm))*(1/count)


@guvectorize(['void(float64[:], int64, float64[:])'], '(n),()->(n)')
def moving_mean(a, window, out):
	"""
	Výpočet klouzavého průměru (Moving mean)

	Input: data   .. vektor dat (np.array() "1D")
		   window .. časový úsek, na kterém je počítán mm " (int)

	Output: out .. vypočtený klouzavý průměr pro všechny hodnoty
				   z vektrou data(np.array() "1D")
	"""
	asum = 0.0
	count = 0
	for i in range(window):
		asum += a[i]
		count += 1
		out[i] = asum / count
	for i in range(window, len(a)):
		asum += a[i] - a[i - window]
		out[i] = asum / count


@guvectorize(['void(float64[:], int64, float64[:])'], '(n),()->(n)')
def moving_variance(data, window, mvar):
	"""
	Výpočet klouzavého rozptylu

	Input: data   .. vektor dat (np.array() "1D")
		   window .. časový úsek, na kterém je počítán emm " (int)

	Output: mvar .. vypočtený klouzavý rozptyl pro všechny hodnoty
					z vektrou data(np.array() "1D")
	"""
	mm = moving_mean(data, window)
	dolni_index = 0
	count = 0
	for i in range(window):
		count += 1
		mvar[i] = sum((data[dolni_index: i + 1] - mm[i])**2)*(1/count)
	for i in range(window, len(data)):
		dolni_index = i + 1 - window
		mvar[i] = sum((data[dolni_index: i + 1] - mm[i])**2)*(1/count)


def savitzky_golay_filter(data, window, polyorder, pos_back=1, deriv=0, axis=-1,
						  mode='nearest'):
	"""
	Výpočet Savitzky-Golay filtru - aproximace klouzavého okna (hodnoty uvnitř)
									pomocí konvoluce s polynomeme

	Input: data      .. vektor dat (np.array() "1D")
		   window    .. časový úsek, na kterém je počítán SG filtr " (int)
		   polyorder .. řád polynomu, který je využit při vyhlazování dat v okně
						(int)
		   pos_back  .. je pozice od konce okna, ve níž probíhá aproximace,
						posunem pozice ze středu okna přicházíme o robustnost
						(int)

	Output: output .. data vyhlazená pomocí S-G filtru (np.array() "1D")
	"""
	if pos_back > window:
		raise ValueError("pozice není uvnitř okna")

	#okraje mám defaulte pomocí nearest => nakopíruje krajní body
	if mode not in ["mirror", "nearest", "wrap"]:
		raise ValueError("mode must be 'mirror', 'nearest' or 'wrap'")

	data = np.asarray(data)
	# Nastavli jsem, aby se koeficienty počítaly v posledním bodě -> pos = window_lenght-1
	coeffs = savgol_coeffs(window, polyorder, pos=window - pos_back, deriv=deriv)
	# dále používám stejnou konvoluci jako je v originále
	output = convolve1d(data, coeffs, axis=axis, mode=mode, cval=0.0)

	return output

"""
	Preprocessing
"""

def Preprocessing(data, pocet_stavu, pocet_feature, labels):
	"""
	Funkce předpočítává střední hodnoty a kovarianční matice potřebné pro správné
		fungování modifikovaného HMM

	Input: data          ... data tvaru matice, jejíž sloupce odpovídají jednotlivým
							 příznakům, kde každý řádek odpovídá jednomu pozorování
							 X_n (np.matrix())
		   pocet_stavu   ... integer udávající počet stavů (int)
		   pocet_feature ... integer udávající počet příznaků (int)
		   labels        ... je vektor skutečných stavů, na základě těchoto hodnot
							 jsou data tříděny do skupin

	Output: output       ... je list obsahující matici středních hodot a vícedimenzionální
							 matice "kovariančních matic" ([np.matrix(), np.array((i,j,k))])
	"""
	if np.shape(data)[0] < np.shape(data)[1]:
		raise TypeError("data nemají správný formát")

	#print("pocet rysu = ", pocet_feature, "tvar labels = ",
	#np.shape(labels), "tvar dat: ", np.shape(data))

	sorted_data_according_states = {}

	for state in range(pocet_stavu):
		sorted_data_according_states[state] = {}
		for feature in range(pocet_feature):
			sorted_data_according_states[state][feature] = []

	for label, _ in enumerate(labels): #label, _ in enumerate(labels)
		for feature in range(pocet_feature):
			sorted_data_according_states[labels[label]][feature].append(data[:, feature][label])

	means = np.zeros((pocet_stavu, pocet_feature))
	for i in sorted_data_according_states:
		for j in sorted_data_according_states[i]:
			means[i, j] = np.mean(sorted_data_according_states[i][j])

	variance = np.zeros((pocet_stavu, pocet_feature, pocet_feature))
	for i in sorted_data_according_states:
		for j in sorted_data_according_states[i]:
			variance[i, j, j] = np.var(sorted_data_according_states[i][j])

	return [means, variance]


def normalization(data, delka_useku=20, training_set=True):
	"""

	"""
	if training_set:
		return data/np.mean(data[:delka_useku])
	else:
		return data[delka_useku:]/np.mean(data[:delka_useku])


def Set_Noise(data, velikost_sumu = 1/40):
	noise = np.random.randn(len(data))
	return data + noise * velikost_sumu


def get_num_features(direc):
	"""
	Tato funkce je pomocná k prepare_features a sklouží předběžnému sečtení počtu příznaků

	Input: direc        ... slovník (directory) se všemy potřebnými příznaky jako klíči a jejich konfiguracemi
							jako hodnotami (items)

	Output: length      ... dékla resp. počet příznaků
	"""
	length = 0
	for i in direc.keys():
		if i in ["1.d SGF", "2.d SGF"]:
			# na "H_alpha" se neprám, protože ho musím s přidávat vždy (kvůli vstacku)
			if direc[i] == True:
				length += 1
		elif i == "H_alpha":
			continue
		else:
			length += len(direc[i])
	return length


def make_matrix(data, combin, okna, H_alpha=True):
	"""
	Funkce počítá vybrané příznaky a vytvářní z nich matici vhodnou pro trénink a predikci.
	Tato funkce je schopna počítat jen příznaky se spec zadáním. Pro MM a EMM počítá stejná
	okna. Nezle pro ně počítat rozlišná.

	Tuto funkci od 11.2.2019 nahrazuje funkce prepare_features, kvůli starším verzím kódu
	v některých skriptech je zatím ponechána.
	"""
	if combin != "all" and len(combin) != 5:
		raise ValueError("Chybý nějáky parametr potřebný pro make matrix!")
	if not isinstance(okna, list) or len(okna) < 4:
		raise ValueError("okna musí být list s nejméně 4 prvky")
	if not isinstance(data, list):
		data = [data]

	if combin == "all":
		combin = np.ones(5, dtype=int)

	W = 3 + 2*len(okna[0]) + 2*len(okna[1]) + 2*len(okna[2]) + len(okna[3])
	out = np.zeros((W, 1))
	for d in data:
		norm_d = normalization(d[1], delka_useku=20, training_set=True)
		mat = np.asarray(norm_d)
		if combin[0]:
			mat = np.vstack((mat, savitzky_golay_filter(norm_d, 9, 2, pos_back=5, deriv=1)))
		if combin[1]:
			mat = np.vstack((mat, savitzky_golay_filter(norm_d, 9, 2, pos_back=5, deriv=2)))
		if combin[2]:
			for mm0 in okna[0]:
				mat = np.vstack((mat, moving_mean(norm_d, mm0)))
			for mm1 in okna[1]:
				mat = np.vstack((mat, moving_mean(norm_d, mm1)))
			for mm2 in okna[2]:
				mat = np.vstack((mat, moving_mean(norm_d, mm2)))
		if combin[3]:
			for emm0 in okna[0]:
				mat = np.vstack((mat, exp_moving_mean(norm_d, emm0)))
			for emm1 in okna[1]:
				mat = np.vstack((mat, exp_moving_mean(norm_d, emm1)))
			for emm2 in okna[2]:
				mat = np.vstack((mat, exp_moving_mean(norm_d, emm2)))
		if combin[4]:
			for mv0 in okna[3]:
				mat = np.vstack((mat, moving_variance(norm_d, mv0)))
			#print("W = ", W, "mat = ", np.shape(mat))
		out = np.hstack((out, mat))
	if not H_alpha:
		return out[1:, 1:].T
	else:
		return out[:, 1:].T


def prepare_features(data, config, normalize=True):
	"""
	Funkce počítá vybrané příznaky a vytvářní z nich matici (resp. pd.DataFrame) vhodnou pro trénink a predikci

	Input: data          ... data tvaru seznamu vektorů (list of arrays).
							 Jedná se o seznam jednotlivých signálů H_alpha, ze kterých se pak
							 počítají všechny příznaky. Do jedné společné matice se spojí až tady.
							 je to z důvodů správného výpočtu příznaků (kvůli správným
							 vypočtům příznaků na počátečních hodnotách signálu)
							 Správný "formát" pro vstup, rovnou připravuje funkce load_datasets z
							 modulu datasets(.py)

		   config        ... konfigurece resp. příznaky které chci počítat z dat spolu s parametry.
							 Konfigurace je požadována ve formě slovníku (directory), kde klíče jsou
							 zkratky příznaků a items jsou parametry.

	Output: output       ... je matice resp. dataframe vypočítaných příznaků (pd.DataFrame)

	-------------------------
	Vzor vložení konfigurace:
	-------------------------
	config = {"H_alpha": True, "1.d SGF": True, "2.d SGF": True, "MM": [3,5,7], "EMM": [9,10,11], "MV": [4,7,10,15]}

	kde H_alpha ... je originální signál
		1.d SGF ... je první derivace pomocí Savitzky-Golay filtru
		2.d SGF ... je druhá derivace pomocí Savitzky-Golay filtru
		MM      ... je klouzavý průměr (Moving Mean)
		EMM     ... je exponenciálně tlumený klouzavý průměr (Exponencial Moving Mean)
		MV      ... je klouzavý rozptyl (Moving Variance)
	"""
	output = np.zeros((get_num_features(direc=config)+1, 1))

	for signal in data:
		# Normalizace
		if normalize:
			norm_d = normalization(signal[1], delka_useku=20, training_set=True)
		else:
			norm_d = signal
		mat = np.asarray(norm_d)
		if "1.d SGF" in config.keys() and config["1.d SGF"]==True:
			mat = np.vstack((mat, savitzky_golay_filter(norm_d, 9, 2, pos_back=5, deriv=1)))

		if "2.d SGF" in config.keys() and config["2.d SGF"]==True:
			mat = np.vstack((mat, savitzky_golay_filter(norm_d, 9, 2, pos_back=5, deriv=2)))

		if "MM" in config.keys():
			for okno in config["MM"]:
				mat = np.vstack((mat, moving_mean(norm_d, okno)))

		if "EMM" in config.keys():
			for okno in config["EMM"]:
				mat = np.vstack((mat, exp_moving_mean(norm_d, okno)))

		if "MV" in config.keys():
			for okno in config["MV"]:
				mat = np.vstack((mat, moving_variance(norm_d, okno)))

			#print("W = ", W, "mat = ", np.shape(mat))
		output = np.hstack((output, mat))
	if "H_alpha" not in config.keys() or config["H_alpha"] !=True:
		return output[1:, 1:].T
	else:
		return output[:, 1:].T


class Features:
	def __init__(self, config, normalize):
		"""
		config musí mít formu slovníku, a je v něm pouze konfigurace příznaků a jejich oken
		"""
		self.config = config
		self.normalize = normalize
		self.savgol_params = Bunch(window=9, polyorder=2, pos_back=5)


	@classmethod
	def all_features(cls):
		CONFIG = {"H_alpha": True,
				  "1.d SGF": True,
				  "2.d SGF": True,
				  "MM": [4,6,8,10,12,14,16],
				  "EMM": [4,6,8,10,12,14,16],
				  "MV": [5,6,7,8,9,10,11,12,13,14,15,16]}
		return cls(CONFIG, True)


	def get_names(self, labels=False):
		"""
		Fce vrací názvy příznaků a případně i labely jako seznam.
			(vhodný jako názvy sloupců v dataframu)
		"""
		names = []
		if self.config["H_alpha"]:
			names.append("H_alpha")
		if self.config["1.d SGF"]:
			names.append("1.d SGF")
		if self.config["2.d SGF"]:
			names.append("2.d SGF")
		for i in self.config["MM"]:
			names.append(f"MM {i}")
		for i in self.config["EMM"]:
			names.append(f"EMM {i}")
		for i in self.config["MV"]:
			names.append(f"MV {i}")
		if labels:
			names.append("labels")
		return names


	def fit_transform(self, Data, diffType=False):
		"""
		@upreavená fce prepare_features
		Funkce počítá vybrané příznaky a vytvářní z nich matici (resp. pd.DataFrame) vhodnou pro trénink a predikci

		Input: data          ... data tvaru seznamu vektorů (list of arrays).
								 Jedná se o seznam jednotlivých signálů H_alpha, ze kterých se pak
								 počítají všechny příznaky. Do jedné společné matice se spojí až tady.
								 je to z důvodů správného výpočtu příznaků (kvůli správným
								 vypočtům příznaků na počátečních hodnotách signálu)
								 Správný "formát" pro vstup, rovnou připravuje funkce load_datasets.H_alpha z
								 modulu datasets(.py)
				diffType	 ... Pokud nevkládám pouze .H_alpha (list of arrays), ale .data

		Private: config      ... konfigurece resp. příznaky které chci počítat z dat spolu s parametry.
								 Konfigurace je požadována ve formě slovníku (directory), kde klíče jsou
								 zkratky příznaků a items jsou parametry.

		Output: output       ... je matice resp. dataframe vypočítaných příznaků (pd.DataFrame)

		-------------------------
		Vzor vložení konfigurace:
		-------------------------
		config = {"H_alpha": True, "1.d SGF": True, "2.d SGF": True, "MM": [3,5,7], "EMM": [9,10,11], "MV": [4,7,10,15]}

		kde H_alpha ... je originální signál
			1.d SGF ... je první derivace pomocí Savitzky-Golay filtru
			2.d SGF ... je druhá derivace pomocí Savitzky-Golay filtru
			MM      ... je klouzavý průměr (Moving Mean)
			EMM     ... je exponenciálně tlumený klouzavý průměr (Exponencial Moving Mean)
			MV      ... je klouzavý rozptyl (Moving Variance)
		"""

		output = np.zeros((get_num_features(direc=self.config)+1, 1))

		for signal in Data:
			if diffType:
				signal = signal[1]

			# Normalizace
			if self.normalize:
				norm_d = normalization(signal, delka_useku=20, training_set=True)
			else:
				norm_d = signal
			mat = np.asarray(norm_d)
			if "1.d SGF" in self.config.keys() and self.config["1.d SGF"]==True:
				mat = np.vstack((mat, savitzky_golay_filter(data=norm_d,
															window=self.savgol_params.window,
															polyorder=self.savgol_params.polyorder,
															pos_back=self.savgol_params.pos_back,
															deriv=1)))

			if "2.d SGF" in self.config.keys() and self.config["2.d SGF"]==True:
				mat = np.vstack((mat, savitzky_golay_filter(data=norm_d,
															window=self.savgol_params.window,
															polyorder=self.savgol_params.polyorder,
															pos_back=self.savgol_params.pos_back,
															deriv=2)))

			if "MM" in self.config.keys():
				for okno in self.config["MM"]:
					mat = np.vstack((mat, moving_mean(norm_d, okno)))

			if "EMM" in self.config.keys():
				for okno in self.config["EMM"]:
					mat = np.vstack((mat, exp_moving_mean(norm_d, okno)))

			if "MV" in self.config.keys():
				for okno in self.config["MV"]:
					mat = np.vstack((mat, moving_variance(norm_d, okno)))

				#print("W = ", W, "mat = ", np.shape(mat))
			output = np.hstack((output, mat))
		if "H_alpha" not in self.config.keys() or self.config["H_alpha"] !=True:
			return output[1:, 1:].T
		else:
			return output[:, 1:].T


	def fit_transform_batch(self, Data):
		"""
		Tady vstupem není list-of-lists, ale jen array
		"""
		return self.fit_transform([Data])


class KFold:
	"""
	Třída vytvořena pro zrychlení práce s trénovacími a testovacími datasety
		- nemusím délky pořád dávat jako parametr do funkce
		- narodzíl od Sklearnu je přímo na míru potřebám
	"""
	def __init__(self, lengths = None):
		if lengths == None:
			raise ValueError("Nezadali jste parametry!!")    
		self.lengths = np.cumsum(lengths)
		
	def fit_transform(self, x, kFoldIndex):
		"""
		Funkce připravuje části pro K-fold crossvalidaci, tzn. vrací k-té fold

		Input:  x           	... pd.DataFrame(), dataframe s příznaky
				kFoldIndex   	... int, index v listu délek (lengths) - výběr k-tého foldu
			   
		Output: train       	... Dataframe bez k-tého foldu
				test        	... Dataframe k-tého foldu
		"""
		if np.shape(x)[0] != np.max(self.lengths):
			raise ValueError("Počet pozorování se neschoduje s součtem délek")
			
		
		if kFoldIndex == 0:
			down = 0
		else:
			down = self.lengths[kFoldIndex-1]
			
		up = self.lengths[kFoldIndex]
		
		train = copy(x.drop(x.index[down:up]))
		test = copy(x[down:up])

		X_train = train.drop(columns=["labels"])
		X_test = test.drop(columns=["labels"])
		y_train = train['labels']
		y_test = test.labels

		return X_train, y_train, X_test, y_test



