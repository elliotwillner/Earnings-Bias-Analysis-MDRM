import numpy as np
import pandas as pd
import pickle
import os
import datetime
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation , Masking, Bidirectional, GlobalAvgPool1D, GlobalMaxPool1D, Conv1D, TimeDistributed, Input, Concatenate, GRU, dot, multiply, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import scipy.stats as stats
import seaborn as sns
# from scipy.stats import pearsonr
# from scipy.stats import spearmanr
from scipy.stats import ttest_ind

def loadPickleFile(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def loadData():
    audio_featDict = loadPickleFile('./data/audio_featDict.pkl')
    audio_featDictMark2 = loadPickleFile('./data/audio_featDictMark2.pkl')
    genders = loadPickleFile('./data/genders.pkl')
    df = pd.read_csv("./data/full_stock_data.csv")
    return audio_featDict, audio_featDictMark2, genders, df

def createLstmMatrix(speaker_list, audio_featDict, audio_featDictMark2, text_file_name):
    temp = np.zeros((520, 26), dtype=np.float64)
    for i, sent in enumerate(speaker_list):
        try:
            temp[i, :] = audio_featDict[text_file_name][sent] + audio_featDictMark2[text_file_name][sent]
        except KeyError:
            continue
    return temp

def modifyData(df, audio_featDict, audio_featDictMark2, genders=None):
    X, y_labels = [], {'male': [], 'female': []}
    errors = []

    for index, row in df.iterrows():
        try:
            speaker_list = sorted(list(audio_featDict[row['text_file_name']].keys()),
                                  key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))
            lstm_matrix = createLstmMatrix(speaker_list, audio_featDict, audio_featDictMark2, row['text_file_name'])
            X.append(lstm_matrix)

            if genders:
                gender_key = 'male' if genders[row['text_file_name']] == 'M' else 'female'
                for days in [3, 7, 15, 30]:
                    y_labels[gender_key].append(float(row[f'future_{days}']))
        except Exception as e:
            errors.append(row['text_file_name'])
            X.append(np.zeros((520, 26), dtype=np.float64))

    X = np.array(X)
    for key in y_labels.keys():
        y_labels[key] = np.array(y_labels[key])

    return X, y_labels, errors

def compareFeat(X_male, X_female, feature_names):
    count = 0
    for i in range(X_male.shape[-1]):
        male_feat = X_male[:, :, i].flatten()
        female_feat = X_female[:, :, i].flatten()
        t, p = ttest_ind(male_feat, female_feat)
        if p < 0.05:
            count += 1
        print(f"{feature_names[i]} : p={p}, t={t}")
    print(f"Count of statistically significant different features: {count}")

#Main script
audio_featDict, audio_featDictMark2, genders, df = loadData()
X, y_labels, errors = modifyData(df, audio_featDict, audio_featDictMark2, genders=genders)

X_male = X[y_labels['male']]
X_female = X[y_labels['female']]

#NaN Value Handling
inds_X = np.where(np.isnan(X))
for i in range(len(inds_X[0])):
    row_mean_X = np.nanmean(X[inds_X[0][i], :, inds_X[2][i]], axis=0)
    X[inds_X[0][i], inds_X[1][i], inds_X[2][i]] = row_mean_X

#Handle NaN values in X_male
inds_X_male = np.where(np.isnan(X_male))
for i in range(len(inds_X_male[0])):
    row_mean_X_male = np.nanmean(X_male[inds_X_male[0][i], :, inds_X_male[2][i]], axis=0)
    X_male[inds_X_male[0][i], inds_X_male[1][i], inds_X_male[2][i]] = row_mean_X_male

#Handle NaN values in X_female
inds_X_female = np.where(np.isnan(X_female))
for i in range(len(inds_X_female[0])):
    row_mean_X_female = np.nanmean(X_female[inds_X_female[0][i], :, inds_X_female[2][i]], axis=0)
    X_female[inds_X_female[0][i], inds_X_female[1][i], inds_X_female[2][i]] = row_mean_X_female


feature_names = ['Mean F0', 'Stdev F0', 'Hnr', 'Local Jitter', 'Local Absolute Jitter', 'Rap Jitter', 'Ppq5 Jitter', 'Ddp Jitter', 'Local Shimmer', 'Localdb Shimmer', 'Apq3 Shimmer', 'Aqpq5 Shimmer', 'Apq11 Shimmer', 'Dda Shimmer', 'N Pulses', 'N Periods', 'Degree Of Voice Breaks', 'Mean Intensity', 'Sd Energy', 'Max Intensity', 'Min Intensity', 'Max Pitch', 'Min Pitch', 'Voiced Frames', 'Voiced To Total Ratio', 'Voiced To Unvoiced Ratio']