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



# Read genders data from CSV file
genders_df = pd.read_csv("./data/genders.csv")

# Convert genders data to a dictionary
genders_dict = genders_df.set_index('folder_name')['gender'].to_dict()

# Save genders dictionary as a pickle file
with open('./data/genders.pkl', 'wb') as f:
    pickle.dump(genders_dict, f)

with open('./data/genders.pkl', 'rb') as f:
    genders=pickle.load(f)