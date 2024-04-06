import numpy as np
import pandas as pd
import pickle
import os
import datetime
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation , Masking, Bidirectional, GlobalAvgPool1D, GlobalMaxPool1D, Conv1D, TimeDistributed, Input, Concatenate, GRU, dot, multiply, concatenate, add, Lambda
from tensorflow.keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import tensorflow as tf

with open('./data/audio_featDict.pkl', 'rb') as f:
    audio_featDict=pickle.load(f)

with open('./data/audio_featDictMark2.pkl', 'rb') as f:
    audio_featDictMark2=pickle.load(f)

## add finbert embeddings here --> change path
with open('./data/finbert_earnings.pkl', 'rb') as f:
    text_dict=pickle.load(f)

with open('./data/genders.pkl', 'rb') as f:
    genders=pickle.load(f)

traindf= pd.read_csv("./data/train_split.csv")
testdf=pd.read_csv("./data/test_split.csv")
valdf=pd.read_csv("./data/val_split.csv")


error=[]
error_text=[]

print(len(text_dict))

def ModifyData(df,text_dict, genders = None):
    X=[]
    X_text=[]
    y_3days=[]
    y_7days=[]
    y_15days=[]
    y_30days=[]

    if not genders is None:
        print('Got genders --', len(genders))
        X_male=[]
        X_female = []
        X_text_male = []
        X_text_female = []
        y_3days_male = []
        y_3days_female = []
        y_7days_male = []
        y_7days_female = []
        y_15days_male = []
        y_15days_female = []
        y_30days_male = []
        y_30days_female = []