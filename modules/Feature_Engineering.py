# Feature_Engineering
# Autor: Matěj Zorek
# Modul slouží k výpočtům příznaků a obecně předzpracování datasetu

"""
TO DO:
	1) figure out how to train batches
			- malé množství přes fit_transform_batch	DONE 14.4.2019
			- větší množství se nejdříve vypočítá 
				buď hromadně nebo přes batch
				a uloží se do h.5 - do tréninku ho 
				budu načítat po částech					DONE 14.4.2019
	2) finish Feature class 							DONE 14.4.2019
	3) classmethod all_features 						DONE 13.4.2019
	4) fix inputs format in prepare_features
		- vytvořil jsem fit_transform, kde je to už
			opravené									DONE 14.4.2019	
	5) add changeable parameters for SGF
		- přidání self.savgol_params umožňuje ve
			třídě měnit i další parametry				DONE 14.4.2019
	6) add function, which creates batches
		- vypočítám příznaky pro každý signal a 
			uložím je zvlášť v DataFrame (tzn KFold)	DONE 26.4.2019
	7) add KFold class 									
		- zrychlení operace s pozorováními				DONE 26.4.2019
"""


import numpy as np
import pandas as pd
from copy import copy
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs
from numba import guvectorize
from bunch import Bunch
from modules.Datasets import merge_labels  # ?

"""
	Fce pro výpočty příznaků
"""


@guvectorize(['void(float64[:], int64, float64[:])'], '(n),()->(n)')
def derivace(data, step, derivation):
	"""
	Výpočet derivace pomocí centrální diference 2. řádu.

	Input: data .. vektor dat (np.array() "1D")
			step .. časový krok "h" (float64)

	Output: derivation .. vypočtená derivace pro všechny hodnoty z vektoru data
							(np.array() "1D")
	"""
	data = np.array(data)
	kon = len(data)-1
	derivation[0] = (data[1] - data[0]) / step
	derivation[kon] = (data[kon] - data[kon - 1]) / step
	for i in range(1, len(data)-1):
		derivation[i] = (data[i + 1] - data[i - 1]) / (2 * step)


@guvectorize(['void(float64[:], int64, float64[:])'], '(n),()->(n)')
def exp_moving_mean(data, window_length, emm):
	"""
	Výpočet exponenciálně tlumeného klouzavého průměru (Exponential moving mean)

	Input:  data   		 	.. vektor dat (np.array() "1D")
			window_length 	.. časový úsek, na kterém je počítán emm " (int)

	Output: emm 			..  vypočtený exponenciální klouzavý průměr pro všechny hodnoty
								z vektoru data(np.array() "1D")
	"""
	gamma1 = 0.9 ** np.arange(window_length)
	gamma = 0.9 ** np.arange(window_length)[::-1]
	count = 0
	bottom_index = 0
	for i in range(window_length):
		count += 1
		emm[i] = (sum(data[bottom_index: i + 1] * gamma1[bottom_index: i + 1][::-1])) * (1/count)
	for i in range(window_length, len(data)):
		bottom_index = i + 1 - window_length
		emm[i] = (sum(data[bottom_index: i + 1] * gamma)) * (1 / count)


@guvectorize(['void(float64[:], int64, float64[:])'], '(n),()->(n)')
def moving_mean(a, window_length, out):
	"""
	Výpočet klouzavého průměru (Moving mean)

	Input:  data   .. vektor dat (np.array() "1D")
			window .. časový úsek, na kterém je počítán mm " (int)

	Output: out ..  vypočtený klouzavý průměr pro všechny hodnoty
					z vektoru data(np.array() "1D")
	"""
	cumulative_sum = 0.0
	count = 0
	for i in range(window_length):
		cumulative_sum += a[i]
		count += 1
		out[i] = cumulative_sum / count
	for i in range(window_length, len(a)):
		cumulative_sum += a[i] - a[i - window_length]
		out[i] = cumulative_sum / count


@guvectorize(['void(float64[:], int64, float64[:])'], '(n),()->(n)')
def moving_variance(data, window_length, mvar):
	"""
	Výpočet klouzavého rozptylu

	Input:  data   .. vektor dat (np.array() "1D")
			window .. časový úsek, na kterém je počítán emm " (int)

	Output: mvar .. vypočtený klouzavý rozptyl pro všechny hodnoty
					z vektoru data(np.array() "1D")
	"""
	mm = moving_mean(data, window_length)
	bottom_index = 0
	count = 0
	for i in range(window_length):
		count += 1
		mvar[i] = sum((data[bottom_index: i + 1] - mm[i])**2)*(1/count)
	for i in range(window_length, len(data)):
		bottom_index = i + 1 - window_length
		mvar[i] = sum((data[bottom_index: i + 1] - mm[i])**2)*(1/count)


def savitzky_golay_filter(data, window, polyorder, pos_back=1, order=0, axis=-1, mode='nearest'):
	"""
	Výpočet Savitzky-Golay filtru - aproximace klouzavého okna (hodnoty uvnitř)
									pomocí konvoluce s polynomem

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

	# okraje mám default pomocí nearest => nakopíruje krajní body
	if mode not in ["mirror", "nearest", "wrap"]:
		raise ValueError("mode must be 'mirror', 'nearest' or 'wrap'")

	data = np.asarray(data)
	# Nastavili jsem, aby se koeficienty počítaly v posledním bodě -> pos = window_lenght-1
	coeffs = savgol_coeffs(window, polyorder, pos=window - pos_back, deriv=order)
	# dále používám stejnou konvoluci jako je v originále
	output = convolve1d(data, coeffs, axis=axis, mode=mode, cval=0.0)

	return output


"""
	Preprocessing
"""


def Preprocessing(data, num_states, num_features, labels):
	"""
	Funkce předpočítává střední hodnoty a kovarianční matice potřebné pro správné
		fungování modifikovaného HMM

	Input: data          ... data tvaru matice, jejíž sloupce odpovídají jednotlivým
								příznakům, kde každý řádek odpovídá jednomu pozorování
								X_n (np.matrix())
			num_states   ... integer udávající počet stavů (int)
			num_features ... integer udávající počet příznaků (int)
			labels       ... je vektor skutečných stavů, na základě těchto hodnot
								jsou data tříděny do skupin

	Output: output       ... je list obsahující matici středních hodnot a vícedimenzionální
							matice "kovariančních matic" ([np.matrix(), np.array((i,j,k))])
	"""
	if np.shape(data)[0] < np.shape(data)[1]:
		raise TypeError("data nemají správný formát")

	# print("pocet rysu = ", pocet_feature, "tvar labels = ",
	# np.shape(labels), "tvar dat: ", np.shape(data))

	sorted_data_according_states = {}

	for state in range(num_states):
		sorted_data_according_states[state] = {}
		for feature in range(num_features):
			sorted_data_according_states[state][feature] = []

	for label, _ in enumerate(labels):  # label, _ in enumerate(labels)
		for feature in range(num_features):
			sorted_data_according_states[labels[label]][feature].append(data[:, feature][label])

	means = np.zeros((num_states, num_features))
	for i in sorted_data_according_states:
		for j in sorted_data_according_states[i]:
			means[i, j] = np.mean(sorted_data_according_states[i][j])

	variance = np.zeros((num_states, num_features, num_features))
	for i in sorted_data_according_states:
		for j in sorted_data_according_states[i]:
			variance[i, j, j] = np.var(sorted_data_according_states[i][j])

	return [means, variance]


def normalization(data, window_length=20, training_set=True):
	"""
	Funkce provádí primitivní škálování signálu.

	:param data: signal typu 1D np.array
	:param window_length: délka okna, podle kterého normalizujeme
	:param training_set: typ signálu (tréninkový vs testovací), při real-aplikacích nebude toto okno klasifikované
	:return: normalizovaný signál
	"""
	if training_set:
		return data/np.mean(data[:window_length])
	else:
		return data[window_length:] / np.mean(data[:window_length])


def set_noise(data, velikost_sumu=1/40):
	noise = np.random.randn(len(data))
	return data + noise * velikost_sumu


def get_num_features(dictionary):
	"""
	Tato funkce je pomocná k prepare_features a slouží předběžnému sečtení počtu příznaků

	Input: dictionary        ... slovník se všemi potřebnými příznaky jako klíči a jejich konfiguracemi
							jako hodnotami (items)

	Output: length      ... délka resp. počet příznaků
	"""
	length = 0
	for i in dictionary.keys():
		if i in ["1.d SGF", "2.d SGF"]:
			# na "H_alpha" se neptám, protože ho musím s přidávat vždy (kvůli vstacku)
			if dictionary[i]:
				length += 1
		elif i == "H_alpha":
			continue
		elif i == "signal" and dictionary[i]:
			length += 1			# kvůli hsf a lsf (více typů signálů)
		else:
			length += len(dictionary[i])
	return length


def make_matrix(data, combin, okna, h_alpha=True):
	"""
	Funkce počítá vybrané příznaky a vytváření z nich matici vhodnou pro trénink a predikci.
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
		norm_d = normalization(d[1], window_length=20, training_set=True)
		mat = np.asarray(norm_d)
		if combin[0]:
			mat = np.vstack((mat, savitzky_golay_filter(norm_d, 9, 2, pos_back=5, order=1)))
		if combin[1]:
			mat = np.vstack((mat, savitzky_golay_filter(norm_d, 9, 2, pos_back=5, order=2)))
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
			# print("W = ", W, "mat = ", np.shape(mat))
		out = np.hstack((out, mat))
	if not h_alpha:
		return out[1:, 1:].T
	else:
		return out[:, 1:].T


def prepare_features(data, config, normalize=True):
	"""
	Funkce počítá vybrané příznaky a vytváření z nich matici (resp. pd.DataFrame) vhodnou pro trénink a predikci

	Input: data          ... 	data tvaru seznamu vektorů (list of arrays).
								Jedná se o seznam jednotlivých signálů H_alpha, ze kterých se pak
								počítají všechny příznaky. Do jedné společné matice se spojí až tady.
								je to z důvodů správného výpočtu příznaků (kvůli správným
								výpočtům příznaků na počátečních hodnotách signálu)
								Správný "formát" pro vstup, rovnou připravuje funkce load_datasets z
								modulu datasets(.py)

			config        ... 	konfigurace resp. příznaky které chci počítat z dat spolu s parametry.
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
	output = np.zeros((get_num_features(dictionary=config) + 1, 1))

	for signal in data:
		# Normalizace
		if normalize:
			norm_d = normalization(signal[1], window_length=20, training_set=True)
		else:
			norm_d = signal
		mat = np.asarray(norm_d)
		if "1.d SGF" in config.keys() and config["1.d SGF"] is True:
			mat = np.vstack((mat, savitzky_golay_filter(norm_d, 9, 2, pos_back=5, order=1)))

		if "2.d SGF" in config.keys() and config["2.d SGF"] is True:
			mat = np.vstack((mat, savitzky_golay_filter(norm_d, 9, 2, pos_back=5, order=2)))

		if "MM" in config.keys():
			for okno in config["MM"]:
				mat = np.vstack((mat, moving_mean(norm_d, okno)))

		if "EMM" in config.keys():
			for okno in config["EMM"]:
				mat = np.vstack((mat, exp_moving_mean(norm_d, okno)))

		if "MV" in config.keys():
			for okno in config["MV"]:
				mat = np.vstack((mat, moving_variance(norm_d, okno)))

			# print("W = ", W, "mat = ", np.shape(mat))
		output = np.hstack((output, mat))
	if "H_alpha" not in config.keys() or config["H_alpha"] is not True:
		return output[1:, 1:].T
	else:
		return output[:, 1:].T


class Features:
	def __init__(self, config, normalize, extended=False):
		"""
		config musí mít formu slovníku, a je v něm pouze konfigurace příznaků a jejich oken
		"""
		self.config = config
		self.normalize = normalize
		self.savgol_params = Bunch(window=9, polyorder=2, pos_back=5)
		self.extended = extended

	@classmethod
	def all_features(cls):
		CONFIG = {
					"H_alpha": True,
					"1.d SGF": True,
					"2.d SGF": True,
					"MM": [4, 6, 8, 10, 12, 14, 16],
					"EMM": [4, 6, 8, 10, 12, 14, 16],
					"MV": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
		}
		return cls(CONFIG, True)

	def get_names(self, labels=False):
		"""
		Fce vrací názvy příznaků a případně i labely jako seznam.
			(vhodný jako názvy sloupců v dataframu)
		"""
		names = []
		if self.config["H_alpha"]:
			names.append("H_alpha")
		if self.extended and self.config["signal"]:
			names.append("signal")
		if self.config["1.d SGF"]:
			names.append("1.d SGF")
		if self.config["2.d SGF"]:
			names.append("2.d SGF")
		if "MM" in self.config.keys():
			for i in self.config["MM"]:
				names.append("MM {}".format(i))
		if "EMM" in self.config.keys():
			for i in self.config["EMM"]:
				names.append("EMM {}".format(i))
		if "MV" in self.config.keys():
			for i in self.config["MV"]:
				names.append("MV {}".format(i))
		if labels:
			names.append("labels")
		return names

	def fit_transform(self, Data, diff_type=False):
		"""
		@ upravená fce prepare_features
		Funkce počítá vybrané příznaky a vytváření z nich matici (resp. pd.DataFrame) vhodnou pro trénink a predikci

		Input: data          ...data tvaru seznamu vektorů (list of arrays).
								Jedná se o seznam jednotlivých signálů H_alpha, ze kterých se pak
								počítají všechny příznaky. Do jedné společné matice se spojí až tady.
								je to z důvodů správného výpočtu příznaků (kvůli správným
								výpočtům příznaků na počátečních hodnotách signálu)
								Správný "formát" pro vstup, rovnou připravuje funkce load_datasets.H_alpha z
								modulu datasets(.py)
				diff_type	 ... Pokud nevkládám pouze .H_alpha (list of arrays), ale .data

		Private: config      ...konfigurace resp. příznaky které chci počítat z dat spolu s parametry.
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
			EMM     ... je exponenciálně tlumený klouzavý průměr (Exponential Moving Mean)
			MV      ... je klouzavý rozptyl (Moving Variance)
		"""

		output = np.zeros((get_num_features(dictionary=self.config) + 1, 1))

		for signal in Data:
			if diff_type:
				signal = signal[1]

			# Normalizace
			if self.normalize:
				norm_d = normalization(signal, window_length=20, training_set=True)
			else:
				norm_d = signal
			mat = np.asarray(norm_d)
			if "1.d SGF" in self.config.keys() and self.config["1.d SGF"] is True:
				mat = np.vstack((mat, savitzky_golay_filter(
															data=norm_d,
															window=self.savgol_params.window,
															polyorder=self.savgol_params.polyorder,
															pos_back=self.savgol_params.pos_back,
															order=1)))

			if "2.d SGF" in self.config.keys() and self.config["2.d SGF"] is True:
				mat = np.vstack((mat, savitzky_golay_filter(
															data=norm_d,
															window=self.savgol_params.window,
															polyorder=self.savgol_params.polyorder,
															pos_back=self.savgol_params.pos_back,
															order=2)))

			if "MM" in self.config.keys():
				for okno in self.config["MM"]:
					mat = np.vstack((mat, moving_mean(norm_d, okno)))

			if "EMM" in self.config.keys():
				for okno in self.config["EMM"]:
					mat = np.vstack((mat, exp_moving_mean(norm_d, okno)))

			if "MV" in self.config.keys():
				for okno in self.config["MV"]:
					mat = np.vstack((mat, moving_variance(norm_d, okno)))

				# print("W = ", W, "mat = ", np.shape(mat))
			output = np.hstack((output, mat))
		if "H_alpha" not in self.config.keys() or self.config["H_alpha"] is not True:
			return output[1:, 1:].T
		else:
			return output[:, 1:].T

	def fit_transform_batch(self, Data):
		"""
		Tady vstupem není list-of-lists, ale jen array
		"""
		return self.fit_transform([Data])

	def CreateDataFrame(self, Data, signal_name=None):
		"""
		Input:  Data 		 ... formát z load_dataset (Bunch)
				config      ... konfigurace

		Output: df           ... dataframe se všemi příznaky

		"""
		X = self.fit_transform(Data=Data.H_alpha)
		lab = merge_labels(labels=Data.labels)

		X = np.hstack((X, lab.reshape(lab.shape[0], 1)))
		cols = self.get_names(labels=True)
		if signal_name is not None:
			cols = ["{}:{}".format(signal_name, name) for name in cols]
		df = pd.DataFrame(data=X, columns=cols)
		return df


class FeatureExtended:
	def __init__(self, config, normalize):
		"""
		:param config: Konfigurace příznaků
						{"H_alpha": {"signal": True,
									"1.d SGF": True,
									"2.d SGF": True,
									"MM": [4,6,8,10,12,14,16],
									"EMM": [4,6,8,10,12,14,16],
									"MV": [5,6,7,8,9,10,11,12,13,14,15,16]},
						"BR_current": {"signal": True,
										"1.d SGF": True,
										"2.d SGF": True,
										"MM": [4,6,8,10,12,14,16],
										"EMM": [4,6,8,10,12,14,16],
										"MV": [5,6,7,8,9,10,11,12,13,14,15,16]},
						...}
		:param normalize: "normalizace" signálů
		"""
		self.config = config
		self.normalize_global = normalize
		self.savgol_params_global = Bunch(window=9, polyorder=2, pos_back=5)
		self.sample_rate = 0  # down-sampling -> data[//self.sample_rate]
		# feature configuration for H_alpha signal
		self.signals = [
			"H_alpha", "BR_current", "BV_current", "MFPS_current", "raw_density", "EFIT_magnetic_axis_z",
			"EFIT_minor_radius", "EFIT_plasma_area", "EFIT_plasma_energy", "EFIT_plasma_value", "q95"
		]

		if not set(self.signals) <= set(self.config):
			raise IndexError("Some feature name is missing in CONFIG.")

		self.feature_configs = {name: config[name] for name in self.signals}

	def fit_transform(self, data):
		# output = np.zeros((get_num_features(dictionary=self.)+1, 1))
		for shot_index, shot_path in enumerate(data):
			pass

	def load_and_fit_transform(self, load_data_class, dataframe=True):
		# tmp = 1 - int(self.signals["H_alpha"].signal)  # kdyby jsem nechtěl H_alpha signál, tak si potřebuji tento sloupec stejně přidat

		# num_features = np.sum([get_num_features(dictionary=i) for i in self.signals]) + 1  # pro lepení pod sebe

		whole_dataset = []

		for shot_index, shot_path in enumerate(load_data_class):
			# load processed data dataframe
			shot_signals = load_data_class.load(shot_index)
			# down-sampling
			if not self.sample_rate:
				shot_signals.dataframe = shot_signals.dataframe[::self.sample_rate]
				shot_signals.labels = shot_signals.labels[::self.sample_rate]

			# signál po signálu do fit_transform
			shot_signals_features = []  # list vypočítaných "příznakových tabulek" pro jednotlivé signály
			for sig in shot_signals.columns():
				# vytvořit třídu Feature pro každý signál z výstřelu
				feat_class = Features(config=self.config[sig], normalize=self.normalize_global, extended=True)
				signal_features = feat_class.CreateDataFrame(Data=shot_signals[sig], signal_name=sig)
				shot_signals_features.append(signal_features)

			shot_signals_features = pd.concat([shot_signals_features], axis=1)
			whole_dataset.append(shot_signals_features)

		whole_dataset = pd.concat([whole_dataset], axis=0)
		if not dataframe:
			return whole_dataset.values

		return whole_dataset







class KFold:
	"""
	Třída vytvořena pro zrychlení práce s trénovacími a testovacími datasety
		- nemusím délky pořád dávat jako parametr do funkce
		- na rozdíl od Sklearnu je přímo na míru potřebám
	"""
	def __init__(self, lengths=None):
		if lengths is None:
			raise ValueError("Nezadali jste parametry!!")    
		self.lengths = np.cumsum(lengths)
		
	def fit_transform(self, x, kFoldIndex):
		"""
		Funkce připravuje části pro K-fold cross-validaci, tzn. vrací k-té fold

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
