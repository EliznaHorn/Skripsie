#########################################################################
# DTW Funtions
##########################################################################

# %% #############
import librosa
import librosa.display
from pydub import AudioSegment
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter

######################
# dynamic programming
######################
def dp(dist_mat):
    """
    Find minimum-cost path through matrix `dist_mat` using dynamic programming.

    The cost of a path is defined as the sum of the matrix entries on that
    path. See the following for details of the algorithm:

    - http://en.wikipedia.org/wiki/Dynamic_time_warping
    - https://www.ee.columbia.edu/~dpwe/resources/matlab/dtw/dp.m

    The notation in the first reference was followed, while Dan Ellis's code
    (second reference) was used to check for correctness. Returns a list of
    path indices and the cost matrix.
    """

    N, M = dist_mat.shape
    
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # Insertion
            i = i - 1
        elif tb_type == 2:
            # Deletion
            j = j - 1
        path.append((i, j))
        

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    return (path[::-1], cost_mat)

#############################################################################
# load chunk
################################################################################
def load_audio_chunk(file_path, start_ms, end_ms):
        chunk = AudioSegment.from_file(file_path)
        #chunk = audio[start_ms:end_ms]
        #convert audio to numpy array to use in librosa later
        chunk = np.array(chunk.get_array_of_samples())
        if chunk.ndim == 2:
            chunk = chunk.mean(axis=1)
        return chunk.astype(np.float32) / (1 << 15)  # Normalize to [-1, 1]

#########################################################################################
# extract MFCC features
############################################################################################
def extract_features(segment, sr, fft, hop):
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40, n_fft=fft, hop_length = hop)
    mfccs_scaled = mfccs.flatten() #np.mean(mfccs.T, axis=0)
    return mfccs_scaled

###############################################################################################
# EXTRACT MEL FEATURES 
##########################################################################################    
def extract_mel_features(segment, sr, fft, hop):
    mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=fft, n_mels = 40, hop_length = hop)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_scaled = mel_db.flatten() #np.mean(mel_db.T, axis=0)
    return mel_scaled 

 
#####################################################################################################  
# Define the band-pass filter design
######################################################################################################
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

##########################################################################################################################   
## WRITE TO TEXTFILE ######
#####################################################################################################################
def write_to_textfile(threshold, normalized_costs, textfile_path):
    datapoint = 0
    selection = 0
    overlapcount = 0
    current_start = None
    current_end = None

    with open(textfile_path, 'x') as file:
        file.write(f"Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\n")
        
        for cost in normalized_costs:
            #If call present, write start time to a textfile
            if cost <= threshold:
                call_start = datapoint * 0.15
                call_end = call_start + 0.3
                
                if current_start is None:
                    current_start = call_start
                    current_end = call_end
                else:
                    if call_start <= current_end and overlapcount <= 1 : #if next call overlaps
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
        
        if current_start is not None:
            selection += 1
            file.write(f"{selection}\t'Spectogram'\t1\t{current_start:.4f}\t{current_end:.4f}\n")


###############################################################################################################           
# Function to calculate overlap duration between two intervals
##################################################################################################################
def overlap_duration(start1, end1, start2, end2):
    return max(0, min(end1, end2) - max(start1, start2))

#####################################################################################################################
# Confusion matrix
#########################################################################################################################
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
            if overlap >= 0.5 * pred_duration:
                matched = True
                break
            
        if not matched:
            FP += 1
        
    # Print the confusion matrix
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")   
    
    return TP, FP, FN
    
