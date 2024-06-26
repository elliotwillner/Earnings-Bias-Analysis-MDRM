import glob
import numpy as np
import pandas as pd
import parselmouth
import os
from tqdm import tqdm
from parselmouth.praat import call
import sklearn
import pickle
import librosa

#Method to call to any praat command
def praatCall(obj, method, *args, **kwargs):
    return parselmouth.praat.call(obj, method, *args, **kwargs)

def measurePitch(voiceID, f0_min, f0_max, unit):
    sound = parselmouth.Sound(voiceID) #reads the sound
    #pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a pitch object
    pitch = sound.to_pitch(f0min = f0_min, f0_max = f0_max) #creates a pitch object

    #audio features
    meanF0 = praatCall(pitch, "Get mean", 0, 0, unit)
    stdevF0 = praatCall(pitch, "Get standard deviation", 0 ,0, unit)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = praatCall(harmonicity, "Get mean", 0, 0)
    pointProcess = praatCall(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)

    #jitters
    jitters = ["local", "local, absolute", "rap", "ppq5", "ddp"]
    jitter_vals = [praatCall(pointProcess, f"Get jitter ({jitter})", 0, 0, 0.0001, 0.02, 1.3) for jitter in jitters]

    #shimmers
    shimmers = ["local", "local_dB", "apq3", "apq5", "apq11", "dda"]
    shimmer_vals = [praatCall([sound, pointProcess], f"Get shimmer ({shimmer})", 0, 0, 0.0001, 0.02, 1.3, 1.6) for shimmer in shimmers]


    pulses = praatCall([sound, pitch], "To PointProcess (cc)")
    n_pulses = praatCall(pulses, "Get number of points")
    n_periods = praatCall(pulses, "Get number of periods", 0.0, 0.0, 0.0001, 0.02, 1.3)

    max_voiced_period = 0.02  #"Longest period" parameter in some of the other queries
    periods = [praatCall(pulses, "Get time from index", i+1) - parselmouth.praat.call(pulses, "Get time from index", i) for i in range(1, n_pulses)]
    degree_of_voice_breaks = sum(p for p in periods if p > max_voiced_period) / sound.duration

    meanIntensity = sound.get_intensity()

    #Aggregate all features into 1 list
    audio_feat = [meanF0, stdevF0, hnr] + jitter_vals + shimmer_vals + [n_pulses, n_periods, degree_of_voice_breaks, meanIntensity]

    return audio_feat


def getProsodicFeat(file_loc):
    unit="Hertz"
    f0_min, f0_max = 75, 300
    sound = parselmouth.Sound(file_loc)
    y, sr = librosa.load(file_loc)
    energy = librosa.feature.rms(y=y)
    SD_energy = np.std(energy)

    #Pitch analysis
    # pitch = sound.to_pitch(f0min=f0_min, f0max=f0_max)
    # maxPitch = pitch.get_maximum(frequency_unit=unit, method='Parabolic')
    # minPitch = pitch.get_minimum(frequency_unit=unit, method='Parabolic')
    pitch = sound.to_pitch(time_step=None, pitch_floor=f0_min, pitch_ceiling=f0_max)

    # Extract pitch points
    pitch_points = pitch.selected_array['frequency']

    # Obtain maximum and minimum pitch values
    maxPitch = np.max(pitch_points)
    minPitch = np.min(pitch_points)

    #Intensity analysis
    intensity = sound.to_intensity(minimum_pitch=f0_min)
    maxIntensity = intensity.get_maximum()
    minIntensity = intensity.get_minimum()
    # maxIntensity = intensity.get_maximum(method='Parabolic')
    # minIntensity = intensity.get_minimum(method='Parabolic')

    #Voiced vs. Total frames analysis
    voiced_frames = pitch.count_voiced_frames()
    total_frames = pitch.get_number_of_frames()
    voiced_to_total_ratio = voiced_frames / total_frames if total_frames else 0
    voiced_to_unvoiced_ratio = voiced_frames / (total_frames - voiced_frames) if total_frames - voiced_frames > 0 else 0


    return [SD_energy, maxIntensity, minIntensity, maxPitch, minPitch, voiced_frames, voiced_to_total_ratio, voiced_to_unvoiced_ratio]


def updateAudioFeat(dataset_path, feature_dict_path, feature_extraction_function, error_list=None):
    folders = os.listdir(dataset_path)

    if error_list is not None:
        folders = error_list

    for folder in tqdm(folders):
        folder_path = os.path.join(dataset_path, folder)

        if not os.path.isdir(folder_path):
            continue

        audio_folder = os.path.join(folder_path, 'Audio')
        if not os.path.isdir(audio_folder):
            continue

        try:
            with open(feature_dict_path, 'rb') as f:
                audio_feat_dict = pickle.load(f)
        except FileNotFoundError:
            audio_feat_dict = {}

        if folder in audio_feat_dict and error_list is None:
            continue

        audio_feat_dict[folder] = {}

        for aud_file in os.listdir(audio_folder):
            audio_path = os.path.join(audio_folder, aud_file)

            if not os.path.isfile(audio_path):
                continue

            aud_file_key = aud_file[:-4]  # Remove file extension

            # Check if audio feature already exists
            if aud_file_key in audio_feat_dict[folder] and len(audio_feat_dict[folder][aud_file_key]) > 0:
                continue

            #print(aud_file_key)

            # Extract audio features
            audio_feat = feature_extraction_function(audio_path)
            audio_feat_dict[folder][aud_file_key] = audio_feat

            # Save the updated dictionary after each file to minimize data loss
            with open(feature_dict_path, 'wb') as f:
                pickle.dump(audio_feat_dict, f)

        print("Earning Call Done!!!")


updateAudioFeat('./OGdataset', './data/audio_featDict.pkl', getProsodicFeat)

#New feature collection
updateAudioFeat('./OGdataset', './data/audio_featDictMark2.pkl', getProsodicFeat)

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('./data/genders.csv')

# Create a dictionary with speaker as keys and gender as values
genders_dict = {}
for _, row in df.iterrows():
    folder_name = row['folder_name']
    gender = row['gender']
    genders_dict[folder_name] = gender

# Save the dictionary as a pickle file
with open('./data/genders.pkl', 'wb') as f:
    pickle.dump(genders_dict, f)

#Error handling if 'error' list occurs somewhere
error = []
updateAudioFeat('./OGdataset', './data/audio_featDictMark2.pkl', getProsodicFeat, error_list=error)

