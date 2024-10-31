# %% #######################################################################################
# SVM Functions
#########################################################################################

#########################################################################################
# Imports
#########################################################################################
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import random
from pydub import AudioSegment
from scipy.signal import butter, lfilter
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#### LOAD AUDIO ##################################################################################################
def load_audio_chunk(file_path, start_ms, end_ms):
        audio = AudioSegment.from_file(file_path)
        chunk = audio[start_ms:end_ms]
        #convert audio to numpy array to use in librosa later
        chunk = np.array(chunk.get_array_of_samples())
        if chunk.ndim == 2:
            chunk = chunk.mean(axis=1)
        return chunk.astype(np.float32) / (1 << 15)  # Normalize to [-1, 1]

#### EXTRACT MFCC FEATURES ########################################################################################
def extract_features(segment, sr):
    n_fft_value = 2048 
    hop_length = 1024
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40, n_fft=n_fft_value, hop_length = hop_length )
    mfccs_scaled = mfccs.flatten() #np.mean(mfccs.T, axis=0)
    return mfccs_scaled

#### EXTRACT MEL FEATURES #########################################################################################    
def extract_mel_features(segment, sr):
    n_fft_value = 2048 
    hop_length = 1024
    mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft_value, n_mels = 40, hop_length = hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_scaled = mel_db.flatten() #np.mean(mel_db.T, axis=0)
    return mel_scaled 
    
#### BAND-PASS FILTER ##############################################################################################
def butter_bandpass(lowcut, highcut, fs, order):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = lfilter(b, a, data)
    return y
    
#### CREATE TRAINING DATAFRAME #####################################################################################
def training_dataframe(file_list, csv_path):
    # Variables
    # lists to collect the features and lables from all the iteraions in the loop
    all_features = []
    all_labels = []

    # Create dataframe
    for j in file_list:
        ##### Load Data #####################
        # Table
        table_path = f'D:\\Universiteit\\5de jaar\\Skripsie Data\\ST-42 Training Selection tables\\selection_table_{j}_ST42.csv'
        table = pd.read_csv(table_path)
        print(f'Table {j} read')

        # Audio File
        audio_path = f'D:\\Universiteit\\5de jaar\\Skripsie Data\\selection_{j}.WAV'

        #get sample rate
        with sf.SoundFile(audio_path) as file:
            sr = file.samplerate

        #### Create Dataset ####################
        features = []
        labels = []
        prevEnd = 0

        #Get the segments with whale call present
        for index, row in table.iterrows():
            startTime = row['Begin Time (s)'] 
            endTime = startTime + 0.3
            label = 1   # whale call is present
            
            #extract segment   (*1000 to work in ms)
            segment = load_audio_chunk(audio_path, startTime*1000, endTime*1000)
            
            # Apply a band-pass filter using librosa's effects module
            filtered_segment = bandpass_filter(segment, 25, 200, sr, 5)
            
            #extract features from the audio
            mfccs_scaled = extract_features(filtered_segment, sr)
            
            features.append(mfccs_scaled)
            labels.append(label)
            
            #to extract no call features between calls
            #extract segments with no whale call
            for i in range(1,2):
                #Get random nocall
                startSegment = random.uniform(prevEnd, startTime-0.2)
                endSegment = startSegment + 0.3
                noCallSegment = load_audio_chunk(audio_path, int(startSegment*1000), int(endSegment*1000))
                
                # Apply a band-pass filter using librosa's effects module
                filtered_noCallSegment = bandpass_filter(noCallSegment, 25, 200, sr, 5)
                
                # extract features
                mfccs_scaled_noCall = extract_features(filtered_noCallSegment, sr)
                
                features.append(mfccs_scaled_noCall)
                labels.append(0)
            
            prevEnd = endTime 
            
        # Collect features and labels from this file
        all_features.extend(features)
        all_labels.extend(labels)

        print(f'Table {j} done') 
        ############################################################################    
    
        
    #convert to dataframe   
    features_df = pd.DataFrame(all_features)
    features_df['Whale call present'] = all_labels

    #Save to csv file
    features_df.to_csv(csv_path, index=False)
    
    return features_df

#### TRAIN MODEL ######################################################################################## 
def train_model(X_train, y_train):
    #define model
    svm = SVC(kernel='rbf')

    # Define the parameter grid for C and gamma
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
    }

    # Set up the grid search
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1', verbose=2, n_jobs=-1)

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Retrieve the best parameters from the grid search
    best_params = grid_search.best_params_

    # Train the model using the best parameters
    svm = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], probability=True, class_weight='balanced')

    # Retrieve the cv_results_ to analyze the performance of each combination
    results = grid_search.cv_results_
    
    # Create a DataFrame to display results
    df_results = pd.DataFrame({
        'C': results['param_C'],
        'gamma': results['param_gamma'],
        'mean_f1': results['mean_test_score']
    })

    # Print the DataFrame sorted by F1 score
    print(df_results.sort_values(by='mean_f1', ascending=False))
    
    return svm.fit(X_train, y_train)
    
#### WRITE TO TEXTFILE ###################################################################
def write_to_textfile(predictions, textfile_path):
    datapoint = 0
    selection = 0
    overlapcount = 0
    current_start = None
    current_end = None

    with open(textfile_path, 'x') as file:
        file.write(f"Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\n")
        
        for prediction in predictions:
            #If call present, write start time to a textfile
            if prediction == 1:
                call_start = datapoint * 0.15
                call_end = call_start + 0.3
                
                if current_start is None:
                    current_start = call_start
                    current_end = call_end
                else:
                    if call_start <= current_end and overlapcount <= 1: #if next call overlaps
                        current_end = max(current_end, call_end)
                        overlapcount = overlapcount + 1
                    else:
                        selection += 1
                        file.write(f"{selection}\t'Spectogram'\t1\t{current_start:.4f}\t{current_end:.4f}\n")
                        overlapcount=0
                        
                        #start new segment
                        current_start = call_start
                        current_end = call_end
            
            #increase datapoint
            datapoint = datapoint + 1
        
        if current_start is not None and prediction == 1:
            selection += 1
            file.write(f"{selection}\t'Spectrogram'\t1\t{current_start:.4f}\t{current_end:.4f}\n")
            
#### CONFUSION MATRIX #########################################################################################
# Function to calculate overlap duration between two intervals
def overlap_duration(start1, end1, start2, end2):
    return max(0, min(end1, end2) - max(start1, start2))
    
def confusion_matrix(ground_truth_path, predicted_path):
    # Read the actual and predicted call tables
    actual_calls = pd.read_csv(ground_truth_path, sep='\t') 
    predicted_calls = pd.read_csv(predicted_path, sep='\t') 

    #convert to numeric
    actual_calls['Begin Time (s)'] = pd.to_numeric(actual_calls['Begin Time (s)'])
    actual_calls['End Time (s)'] = pd.to_numeric(actual_calls['End Time (s)'])
    predicted_calls['Begin Time (s)'] = pd.to_numeric(predicted_calls['Begin Time (s)'])
    predicted_calls['End Time (s)'] = pd.to_numeric(predicted_calls['End Time (s)'])

    ############## determin confusion matrix ######################
    #initialise
    TP = 0
    FP = 0
    FN = 0

    # Kyk na elke actual call en loop dan deur die predicted calls. As daar 'n prediction is wat match , dan TP.
    # As geen prediction match nie, dan is die FN, want model het hom dan nie opgetel nie.
    for _, actual_row in actual_calls.iterrows():
        actual_start = actual_row['Begin Time (s)']
        actual_end = actual_row['End Time (s)']
        matched = False
        
        for _, pred_row in predicted_calls.iterrows():
            pred_start = pred_row['Begin Time (s)']
            pred_end = pred_row['End Time (s)']
            
            #get overlap
            overlap = overlap_duration(actual_start, actual_end, pred_start, pred_end)
            
            #big enough overlap?
            pred_duration = pred_end - pred_start
            if overlap >= 0.5 * pred_duration:
                TP += 1
                matched = True
                break
            
        if not matched:
            FN += 1

            
    # loop deur predictions en vergelyk met actual. As die prediction met geen actual call overlap nie, dan FP        
    for _, pred_row in predicted_calls.iterrows():
        pred_start = pred_row['Begin Time (s)']
        pred_end = pred_row['End Time (s)']
        matched = False
        
        for _, actual_row in actual_calls.iterrows():
            actual_start = actual_row['Begin Time (s)']
            actual_end = actual_row['End Time (s)']
            
            # get overlap
            overlap = overlap_duration(actual_start, actual_end, pred_start, pred_end)
            
            #big enough overlap?
            pred_duration = pred_end - pred_start
            if overlap >= 0.2 * pred_duration:
                matched = True
                break
            
        if not matched:
            FP += 1
            
    return TP, FP, FN
# %%
