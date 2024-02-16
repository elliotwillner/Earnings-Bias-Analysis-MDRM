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
from tensorflow import Callback
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
    audio_featDict = pickle.load(f)
with open('./data/audio_featDictMark2.pkl', 'rb') as f:
    audio_featDictMark2=pickle.load(f)

