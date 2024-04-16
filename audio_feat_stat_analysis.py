import numpy as np
import pandas as pd
import pickle
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
audio_featDict = pickle.load(open('./data/audio_featDict.pkl', 'rb'))
audio_featDictMark2 = pickle.load(open('./data/audio_featDictMark2.pkl', 'rb'))
genders = pickle.load(open('./data/genders.pkl', 'rb'))
df = pd.read_csv('./data/full_stock_data.csv')

# Preprocess data
def extract_features(file_name):
    # Code to extract audio features from dictionaries for the given file_name
    # Return a numpy array containing the feature values
    features = []
    try:
        sentence_features = audio_featDict[file_name]
        for sent_id, feat in sentence_features.items():
            prosodic_feat = audio_featDictMark2[file_name][sent_id]
            features.append(np.concatenate((feat, prosodic_feat)))
    except KeyError:
        print(f"Warning: No features found for file {file_name}")
        return np.zeros((0, 26))  # Return an empty array if file not found

    return np.array(features)

def separate_by_gender(data):
    male_data = []
    female_data = []
    for file_name, features in data.items():
        if genders[file_name] == 'M':
            male_data.append(features)
        else:
            female_data.append(features)
    return np.array(male_data), np.array(female_data)

# Create a dictionary mapping file names to feature arrays
data = {row['text_file_name']: extract_features(row['text_file_name']) for index, row in df.iterrows()}

male_data, female_data = separate_by_gender(data)

# Perform statistical analysis
feature_names = ['Mean F0', 'Stdev F0', 'Hnr', 'Local Jitter', 'Local Absolute Jitter', 'Rap Jitter', 'Ppq5 Jitter', 'Ddp Jitter', 'Local Shimmer', 'Localdb Shimmer', 'Apq3 Shimmer', 'Aqpq5 Shimmer', 'Apq11 Shimmer', 'Dda Shimmer', 'N Pulses', 'N Periods', 'Degree Of Voice Breaks', 'Mean Intensity', 'Sd Energy', 'Max Intensity', 'Min Intensity', 'Max Pitch', 'Min Pitch', 'Voiced Frames', 'Voiced To Total Ratio', 'Voiced To Unvoiced Ratio']

for feature_idx, feature_name in enumerate(feature_names):
    male_feature = male_data[:, feature_idx]
    female_feature = female_data[:, feature_idx]

    # Calculate descriptive statistics
    male_mean = np.mean(male_feature)
    male_std = np.std(male_feature)
    female_mean = np.mean(female_feature)
    female_std = np.std(female_feature)

    # Perform statistical test (e.g., t-test or Mann-Whitney U test)
    t_stat, p_value = ttest_ind(male_feature, female_feature)
    # u_stat, p_value = mannwhitneyu(male_feature, female_feature)

    # Print or store the results
    print(f"{feature_name}: Male (mean={male_mean}, std={male_std}), Female (mean={female_mean}, std={female_std}), p-value={p_value}")

    # Visualize the distributions
    plt.figure()
    sns.histplot(male_feature, kde=True, label='Male', color='b')
    sns.histplot(female_feature, kde=True, label='Female', color='r')
    plt.title(f"{feature_name} Distribution")
    plt.legend()
    plt.show()