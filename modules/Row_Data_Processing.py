# Row_Data_Processing
# Autor: Matěj Zorek
# Modul slouží ke zpracování "surových" dat, získaných z databáze tokamaku COMPASS

import deepdish
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class ProcessedData(object):
    """
    Object for data storing
    """
    def __init__(self, hsf, lsf, info):
        self.hsf = hsf
        self.lsf = lsf
        self.info = info
        self.labels = []

    @classmethod
    def from_preprocessing(cls, data):
        return cls(data[0], data[1], data[2])


class ProcessedAndInterpolatedData(object):

    def __init__(self, processed_data, interpolation_method="linear"):
        self.info = processed_data.info
        self.info["interpolated"] = True
        self.info["interpolation_method"] = interpolation_method
        self.labels = processed_data.labels
        self.dataframe = self.interpole_processed_data(processed_data, interpolation_method)

    @staticmethod
    def interpole_processed_data(processed_data, interpolation_method="linear"):
        if (processed_data.hsf.time.min() != processed_data.lsf.time.min()) and \
           (processed_data.hsf.time.max() != processed_data.lsf.time.max()):
            raise ValueError("Neshoduje se časová osa!!!")

        columns_set = set(processed_data.lsf.columns) - set("time")
        columns_names = list(columns_set)
        interpolation_fs = [interp1d(processed_data.lsf.time.values, processed_data.lsf[col].values, interpolation_method) for col in columns_set]
        lof_interpole = np.array(processed_data.hsf.time.values)
        for i, j in enumerate(columns_set):
            lof_interpole = np.vstack((lof_interpole, interpolation_fs[i](processed_data.hsf.time.values)))
        lof_interpole = lof_interpole[1:, :]
        dataframe = pd.DataFrame(data=lof_interpole.T, columns=columns_names)

        return pd.concat([processed_data.hsf, dataframe], axis=1)

    def save(self, name):
        if ".h5" in name:
            deepdish.io.save("{}".format(name), self)
        else:
            deepdish.io.save("{}.h5".format(name), self)


def save(name, data):
    if ".h5" in name:
        deepdish.io.save("{}".format(name), data)
    else:
        deepdish.io.save("{}.h5".format(name), data)


def check_data(sig):
    indicator = []
    for i, j in enumerate(sig):
        if list(sig[j].keys()) == ["data", "time_axis"]:
            indicator.append(True)
        else:
            indicator.append(False)
            # print(i,j, sig[j].keys())
    return indicator


def correct_time(sig):
    ind = check_data(sig)
    max_start_time = []
    min_end_time = []
    for i, j in enumerate(sig.keys()):
        if ind[i]:
            if np.median(sig[j]["time_axis"]) < 100:
                sig[j]["time_axis"] *= 1000
            max_start_time.append(np.min(sig[j]["time_axis"]))
            min_end_time.append(np.max(sig[j]["time_axis"]))
    return sig, np.max(max_start_time), np.min(min_end_time)


def get_sampling_rates(sig):
    ind = check_data(sig)
    sample_rate = {}
    first_time = {}
    for i, j in enumerate(sig.keys()):
        if ind[i]:
            sample_rate[j] = np.round(np.mean(np.diff(sig[j]["time_axis"])), 4)
            first_time[j] = np.min(sig[j]["time_axis"])
        else:
            sample_rate[j] = "No signal"
            first_time[j] = "No signal"
    return sample_rate, first_time


def cut_data(sig, start, end):
    ind = check_data(sig)
    for i, j in enumerate(sig.keys()):
        if ind[i]:
            mask_start = sig[j]["time_axis"] >= start
            mask_end = sig[j]["time_axis"] <= end
            mask = np.multiply(mask_start, mask_end)
            # print(j, mask)
            sig[j]["time_axis"] = sig[j]["time_axis"][mask]
            sig[j]["data"] = sig[j]["data"][mask]
    return sig


def processing(sig):
    ind = check_data(sig)
    sig, start, end = correct_time(sig)
    sr, ft = get_sampling_rates(sig)
    start += 15  # ms
    end -= 10  # ms
    sig = cut_data(sig, start, end)
    """
    Rozdělím to natvrdo na 0.0005 a 1.0
    v případě jiných frekvencí to nebude fungovat zatím
    """
    feat_names_hsf = []  # high sample frequency
    feat_names_lsf = []  # low sample frequency
    feat_missed = []     # missed values of data
    time_added_hsf = False
    time_added_lsf = False

    for i, j in enumerate(sig.keys()):
        if ind[i]:
            if sr[j] == 0.0005:
                if not time_added_hsf:
                    features_hsf = np.array(sig[j]["time_axis"])
                    feat_names_hsf.append("time")
                    time_added_hsf = True
                feat_names_hsf.append(j)
                if j == "H_alpha":
                    features_hsf = np.vstack((features_hsf, -sig[j]["data"]))
                else:
                    features_hsf = np.vstack((features_hsf, sig[j]["data"]))
            elif sr[j] == 1.0:
                if not time_added_lsf:
                    features_lsf = np.array(sig[j]["time_axis"])
                    feat_names_lsf.append("time")
                    time_added_lsf = True
                feat_names_lsf.append(j)
                features_lsf = np.vstack((features_lsf, sig[j]["data"]))
            else:
                raise ValueError("Something uncommon happend. Last variable: {}, sr: {}".format(j, sr[j]))
        else:
            feat_missed.append(j)
    df_hsf = pd.DataFrame(data=np.transpose(features_hsf), columns=feat_names_hsf)
    df_lsf = pd.DataFrame(data=np.transpose(features_lsf), columns=feat_names_lsf)
    info = {"hsf": feat_names_hsf, "lsf": feat_names_lsf, "missing": feat_missed,
            "time_min": df_hsf.time_axis.min, "time_max": df_hsf.time_axis.max, "hsf_step": 0.0005, "lsf_step": 1.0}

    return df_hsf, df_lsf, info
