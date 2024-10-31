################################
# EXTRACT FEATURES
################################

# %% ################
# Extract 
#####################
# imports
import librosa
import librosa.display
import soundfile as sf
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter

## Functions ################################################################################

def load_audio_chunk(file_path, start_ms, end_ms):
    # Convert milliseconds to seconds
    start_s = start_ms / 1000
    end_s = end_ms / 1000
    
    # Load audio chunk
    audio_chunk, sr = librosa.load(file_path, sr=None, offset=start_s, duration=(end_s - start_s))
    
    # Normalize audio to [-1, 1]
    return audio_chunk.astype(np.float32), sr


def extract_features(segment, sr):
    n_fft_value = 2048  # Adjust n_fft based on segment length
    hop_length = 1024
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40, n_fft=n_fft_value, hop_length = hop_length)
    mfccs_scaled = mfccs.flatten() #np.mean(mfccs.T, axis=0)
    return mfccs_scaled
    
    
def extract_mel_features(segment, sr):
    n_fft_value = 2048  # Adjust n_fft based on segment length
    hop_length = 1024
    mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft_value, n_mels = 40, hop_length = hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_scaled = mel_db.flatten() #np.mean(mel_db.T, axis=0)
    return mel_scaled 
    
# Define the band-pass filter design
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


#############################################################################################
for j in [2,3,4]:
    # file path
    file_path = f'D:\\Universiteit\\5de jaar\\Skripsie Data\\Test Audio\\selection_Test_{j}.WAV'
    
    print(f'Table {j} start')
    
    # convert to dataframe
    detect = []

    with sf.SoundFile(file_path) as file:
        sr = file.samplerate

    segment_size = 300    #0.3s chunks in ms
    chunk_size = 20.1 * 60 * 1000 #20.1 min in ms
    overlap_size = 150  # Overlap size in milliseconds

    start_time = 0
    
    # Process the audio file in chunks
    while True:
        end_time = start_time + chunk_size
        
        if end_time > file.frames / sr * 1000:
            end_time = file.frames / sr * 1000
        
        #extract segment   
        chunk_data, sr = load_audio_chunk(file_path, start_time, end_time)
        
        # Break the loop if no more data is returned (end of file)
        if len(chunk_data) == 0:
            break
        
        current_start = 0
        while current_start + segment_size <= len(chunk_data) / sr * 1000:
            current_end = current_start + segment_size
        
            # Extract the 0.3-second segment
            segment_data = chunk_data[int(current_start / 1000 * sr): int(current_end / 1000 * sr)]
        
            # Apply a band-pass filter using librosa's effects module
            filtered_segment = bandpass_filter(segment_data, 25, 200, sr, 5)

            #extract features from the audio
            mfccs_scaled_detect = extract_features(filtered_segment, sr)
        
            detect.append(mfccs_scaled_detect)
        
            # Move to the next chunk
            current_start += (segment_size - overlap_size)
            #print(f'Moving to segment {current_start/150}')
            
        start_time += chunk_size - overlap_size
        print(f'Done with chunk {start_time/chunk_size}')
        
    #convert to dataframe   
    detect_df = pd.DataFrame(detect)

    #print(detect_df.head())

    excel_file_path = f'D:\\Universiteit\\5de jaar\\Skripsie Data\\DTW\\Test feature tables\\MFCC\\MFCCDetectFeatures_table_Test_{j}_2048_1024.csv'
    detect_df.to_csv(excel_file_path, index=False)
    
    print(f'Table {j} done')

# %%
