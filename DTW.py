#####################################################
# DTW
#####################################################

# %% #################
# Imports
######################
from DTW_functions import load_audio_chunk
from DTW_functions import dp
from DTW_functions import extract_features
from DTW_functions import extract_mel_features
from DTW_functions import bandpass_filter
from DTW_functions import write_to_textfile
from DTW_functions import confusion_matrix

import soundfile as sf
import pandas as pd
import numpy as np
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# %% ###################
# reference template
# ######################

reference_template_path = f'D:\\Universiteit\\5de jaar\\Skripsie Data\\DTW\\Combined template 4\\temp1.WAV'

with sf.SoundFile(reference_template_path) as file:
        sr_ref = file.samplerate
        print("sample ref: ", sr_ref)
    
reference = load_audio_chunk(reference_template_path, 0, 300)

# Apply a band-pass filter using librosa's effects module
filtered_ref = bandpass_filter(reference, 25, 200, sr_ref, 5)

#extract features
reference_features = extract_mel_features(filtered_ref, sr_ref, 2048, 512)

#make 2D numpy array
ref_seq = reference_features.reshape(1, -1)
print("ref seq:", ref_seq)
print(ref_seq.shape)

#scale feature vector
scaler = StandardScaler()
scaler.fit(ref_seq)
ref_seq = scaler.transform(ref_seq)
print("ref seq:", ref_seq)

#print("Norm REF: ", ref_seq)
# %% ###################
# Combine template
# ######################
aligned_templates = np.empty((0,200))

for j in range(2,3):
    template_path = f'D:\\Universiteit\\5de jaar\\Skripsie Data\\DTW\\Combined template 4\\temp{j}.WAV'
    
    with sf.SoundFile(template_path) as file:
        sr_template = file.samplerate
    
    # load audio
    template = load_audio_chunk(template_path, 0, 300)
    
    # Apply a band-pass filter using librosa's effects module
    filtered_template = bandpass_filter(template, 25, 200, sr_template, 5)
    
    # extract features
    template_features = extract_mel_features(filtered_template, sr_template, 2048, 512)
    
    # reshape
    temp_seq = template_features.reshape(1, -1)
    print(j)
    print("temp seq:", temp_seq)
    print(temp_seq.shape)
    #scale features
    temp_seq = scaler.transform(temp_seq)
    #print("Norm temp: {j}: ", temp_seq)

    aligned_templates = np.vstack([aligned_templates, temp_seq])  

aligned_templates = np.vstack([aligned_templates, ref_seq])       
print(aligned_templates)
print(aligned_templates.shape)

#combined
combined_template = np.mean(aligned_templates, axis = 0)

# Ensure combined template is 2D (reshaped if necessary)
combined_template = combined_template.reshape(1, -1)
print(combined_template.shape)

#convert to dataframe   
combined_template_df = pd.DataFrame(combined_template)
print(combined_template_df)

# %%#################
# Plot 
####################
# Reshape to a 2D array (e.g., 1x40 for visualization)
combined_template_2d = combined_template_df

#plot combined
fig, ax = plt.subplots(figsize=(9, 5))
cax = ax.imshow(combined_template_2d, origin="lower", aspect='auto', interpolation="nearest")
ax.set_yticks([])
plt.colorbar(cax, ax=ax)
plt.xlabel("Features")
plt.title("Combined Template Feature Visualisation")
plt.show()

# %% ######################
# Get Feature weights
###########################
from sklearn.ensemble import RandomForestClassifier
X_train = []
y_train = []

for i in range(1, 23):
    train_path = f'D:\\Universiteit\\5de jaar\\Skripsie Data\\DTW\\Feature weights\\Call\\temp{i}.WAV'
    
    with sf.SoundFile(train_path) as file:
        sr_train = file.samplerate
    
    train = load_audio_chunk(train_path, 0, 300)
    train_filtered = bandpass_filter(train, 25, 200, sr_train, 5)
    train_features = extract_mel_features(train_filtered, sr_train, 2048, 512)
   
    print(i)
    print(train_features)
    print(train_features.shape)
    
    X_train.append(train_features)
    y_train.append(1)
 
for i in range(1, 23):
    train_path = f'D:\\Universiteit\\5de jaar\\Skripsie Data\\DTW\\Feature weights\\NoCall\\temp{i}.WAV'
    
    with sf.SoundFile(train_path) as file:
        sr_train = file.samplerate
    
    train = load_audio_chunk(train_path, 0, 300)
    train_filtered = bandpass_filter(train, 25, 200, sr_train, 5)
    train_features = extract_mel_features(train_filtered, sr_train, 2048, 512)
    
    X_train.append(train_features) 
    y_train.append(0)  

print(X_train)
print(y_train)

X_df = pd.DataFrame(X_train)
X_df_scaled = scaler.transform(X_df)

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_df_scaled, y_train)

# Get feature importance scores
importance_scores = model.feature_importances_

# Normalize importance scores so they sum to 1 
importance_weights = importance_scores / np.sum(importance_scores)

print("Feature importance weights:", importance_weights)

indices = np.argsort(importance_weights)[::-1]
top_n = 50
top_features_indices = indices[:top_n]

# %%#################
# Plot 
####################
weighted_combined = combined_template * importance_weights
weighted_combined_df = pd.DataFrame(weighted_combined)


#plot combined
fig, ax = plt.subplots(figsize=(9, 5))
cax = ax.imshow(weighted_combined_df, origin="lower", aspect='auto', interpolation="nearest")
ax.set_yticks([])
plt.colorbar(cax, ax=ax)
plt.xlabel("Features")
plt.title("Combined Template weighted feature Visualisation")
plt.show()


# %% ######################
# load test in as dataframe
############################
table_path = f'D:\\Universiteit\\5de jaar\\Skripsie Data\\ST-42 Feature tables\\Mel\\MelDetectFeatures_table_9_2048_512.csv'
detect_df = pd.read_csv(table_path)
print(detect_df.head())


##################
# Loop through test
######################
normalized_costs = []
costs = []

#x_seq = ref_seq * importance_weights
x_seq = combined_template * importance_weights
#x_seq = x_seq[:, :100]
print(x_seq.shape)

for index, row in detect_df.iterrows():
    print("Reading chunk ", index)
    # reshape
    y_seq = row.values.reshape(1, -1)
    print(y_seq.shape)
    
    # scale
    y_seq_norm = scaler.transform(y_seq)
    
    #weighted
    y_seq_weighted = y_seq_norm * importance_weights
    #y_seq_reduced = y_seq_norm[:, :100]
    #print(y_seq_reduced.shape)
    
    # dtw
    dist_mat = dist.cdist(x_seq, y_seq_weighted, "cosine")
    path, cost_mat = dp(dist_mat)
    cost = cost_mat[-1, -1]
    costs.append(cost)
    print("Alignment cost: {:.4f}".format(cost))
    
    M = y_seq.shape[0]
    N = x_seq.shape[0]
    normalized_cost = cost_mat[-1, -1] / (M + N)
    normalized_costs.append(normalized_cost)
    
    print("Normalized alignment cost: {:.8f}".format( normalized_cost )) 
    print()
    
 
# %% #################
# Plot cost
##################### 
   
# Create a list of chunk indices (or time points) for the x-axis
chunk_indices = range(len(normalized_costs))

# Plot the normalized alignment costs
plt.figure(figsize=(100, 50))
plt.plot(chunk_indices, normalized_costs, marker='o', linestyle='-', color='b')

# Adjust the intervals on the x-axis and y-axis
plt.xticks(np.arange(0, 24000, 2000))  # More intervals on the x-axis
plt.yticks(np.arange(0, 1, 0.1))  # More intervals on the y-axis

# label axis
plt.xlabel('Segment Index', fontsize = 150)
plt.ylabel('Normalized Alignment Cost', fontsize = 150)
plt.title('Normalized Alignment Cost Across Audio Segments', fontsize = 150)
plt.grid(True)
plt.tight_layout()
plt.show()

# %% ###################
# Get best threshold
########################
recalls = []
precisions = []
thresholds = []
f1_scores= []
df = []
i = 0

for threshold in np.arange(0.65, 0.66, 0.01):
    thresholds.append(threshold)
    print(f'starting with threshold {threshold}')
    
    #table paths
    ground_truth_path = f'D:\\Universiteit\\5de jaar\\Skripsie Data\\ST-42 Testing Selection tables\\selection_table_Test_2_3.txt'
    predicted_path = f'D:\\Universiteit\\5de jaar\\Skripsie Data\\DTW\\Textfiles\\Mel\\Model 10\\test_2_th_{threshold}_2.txt'
    
    #write to textfile with this threshold
    write_to_textfile(threshold, normalized_costs, predicted_path)
    
    #get confusion matrix
    TP, FP, FN = confusion_matrix(ground_truth_path, predicted_path)

    # Calculate Precision
    if (TP + FP) > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0  # Avoid division by zero
     
        
    precisions.append(precision)

    # Calculate Recall
    if (TP + FN) > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0  # Avoid division by zero
        
    recalls.append(recall)
    
    # Calculate F1 score
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    
    f1_scores.append(f1_score)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1_score:.4f}")
    
    i = i+0.5

# %% #######################################################################################
# Plot Precision vs. Recall
##############################################################################################
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, marker='o', linestyle='-', color='b', label='Precision-Recall Curve')

# Annotate each point with the corresponding threshold value
for i, threshold in enumerate(thresholds):
    plt.annotate(f'{threshold:.5f}', (recalls[i], precisions[i]), textcoords="offset points", xytext=(0, 5), ha='center')

best_index = np.argmax(f1_scores)
plt.scatter(recalls[best_index], precisions[best_index], color='red', s=100, edgecolor='k', label='Best Balance')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve with Thresholds')
plt.legend()
plt.grid()
plt.show()


