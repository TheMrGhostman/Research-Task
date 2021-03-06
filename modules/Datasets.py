import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import chain
from bunch import Bunch
from copy import copy
import deepdish

#univerzálně a tak trochu natvrdo zadané umístění
path = os.getcwd()
main_folder_path = path.replace(path.split('/')[-1],'')
path = main_folder_path + "Data_npy/Chosen ones/"

def shape(data):
    """
    náhrada np.shape schopná zmapovat a spočítat všechny rozsahy (shape)
    jednotlivých vrsetv struktury v níž načítám a používám datasety (resp. signály)
    - hlavně je schopna poradit si s "lists in list"
    """
    signal_layers = []
    datapoints = []
    for signal in data:
        signal_layers.append(len(signal))
        if len(np.shape(signal)) != 1:
            datapoints.append(len(signal[0]))

    if len(np.unique(signal_layers)) == 1:
        return((len(data), len(data[0]), datapoints, sum(datapoints)))
    else:
        return((len(data), signal_layers, datapoints, sum(datapoints)))

def get_layer(data, layer):
    output = []
    for signal in data:
        output.append(signal[layer])
    return output


def vizualize(bunch):
    plt.figure("Vyobrazení celého datasetu")
    posun = 0
    for i in bunch.H_alpha:
        plt.plot(np.arange(len(i))+ posun, i, '-', lw=0.5)
        posun = posun + len(i)
    plt.xlabel("x - points count (not time)")
    plt.ylabel(r"$H_{\alpha}$")
    plt.show()

def merge_labels(labels):
    """
    Funkce spojí list labelů do jednoho vektoru, který obsahuje všenhy

    Input: labels        ... data tvaru list vektorů (list of arrays)

    Output: output       ... vektor všech labelů
    """
    output = labels[0]
    for lab in labels[1:]:
        output = np.hstack((output, lab))
    return output

def load_dataset(name, dataset_only=False):
    """
    Snaha o zjednodušení načítání dat pro výpočty.
    """

    signals = np.load(main_folder_path + "/Data_npy/{}.npy".format(name)).tolist()

    dataset = [np.load(path + data) for data in chain(signals)]

    time = get_layer(dataset, 0)
    H_alpha = get_layer(dataset, 1)
    labels = get_layer(dataset, 2)

    return Bunch(data=dataset, time=time, H_alpha=H_alpha, labels=labels,
                 signals=signals, shape=shape(dataset))


def load_processed_data(name, dataset_path=False):
    if not dataset_path:
        names = np.load(main_folder_path + "Data_npy/{}".format(name)).tolist()
    else:
        names = np.load(dataset_path).tolist()

    loaded_data = [deepdish.io.load(path + name) for name in chain(names)]

    return loaded_data

class LoadLargeData:
    def __init__(self, name, dataset_path=False, verbose=True):
        if not dataset_path:
            self.names = np.load(main_folder_path + "Data_npy/{}".format(name)).tolist()
        else:
            self.names = np.load(dataset_path).tolist()
        self.number_of_shots = len(self.names)
        if verbose:
            print("Number of shots: {}".format(self.number_of_shots))
            for i,j in enumerate(self.names):
                print("{} of {}: {}".format(i, self.number_of_shots-1, j))

    def load(self, shot_index):
        if shot_index >= self.number_of_shots:
            raise IndexError("Shot index out of bounds")
        return deepdish.io.load(path + self.names[shot_index])



def PrepareCrossFold(data):
    """
    Funkce připravuje data pro použití při křížové validaci po signálech.
    Použitý poměr je 9:1 (trénink:testování). Vrací data ve formě, kterou
    vyžaduje funkce Classification.validace_hromadna().

    Output: Vrací dva seznamy -> train ... trénovací data
                                 test  ... testovací data
    """
    train = np.empty([10]).tolist()
    test = np.empty([10]).tolist()

    for i, j in enumerate(data):
        if i == 0:
            train[i] = copy(data[:-1])
            test[i] = [copy(data[-1])]
        else:
            train[i] = copy(data[:-(i+1)]) + copy(data[-i:])
            test[i] = [copy(data[-(i+1)])]
    return train, test




def unit_test():
    print(os.getcwd()[:-8])
    print(path)
    dat = load_dataset(name="first_dataset")
    #print(dat.data)
    print(shape(dat.H_alpha))
    print(shape(dat.labels))
    print(dat.shape)
    print(np.shape(merge_labels(dat.labels)))
    vizualize(dat)

    #print(shape(dat.train()))


#unit_test()
