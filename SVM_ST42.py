##################################################################
##################################################################
##  SVM ST42
##################################################################
##################################################################
# %% #################################
# Imports
####################################
from SVM_functions import training_dataframe
from SVM_functions import training_dataframe
from SVM_functions import train_model
from SVM_functions import write_to_textfile
from SVM_functions import confusion_matrix

import pandas as pd
import numpy as np
   
# %% ###################################
# Create dataframe for training
#########################################
file_list = [7,12,13]
csv_file_path = 'D:\\Universiteit\\5de jaar\\Skripsie Data\\ST-42 Training Feature tables\\MFCC\\MFCCFeatures_table_training_3_2048_1024.csv'
features_df1 = training_dataframe(file_list, csv_file_path)
print(features_df1.head())


# %% #####################################
# Import existing dataframe for training
########################################## 
features_table_path = f'D:\\Universiteit\\5de jaar\\Skripsie Data\\ST-42 Training Feature tables\\Mel\\MelFeatures_table_training_3_1024_512.csv'
features_df = pd.read_csv(features_table_path)
print(features_df.head())

# %% ##########################
# Preprocessing
###############################
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

######### Get Dependent Variable #################
X = features_df.drop('Whale call present', axis=1)
y = features_df['Whale call present']

####### Split the data ############################
#create 4 variables returned by this function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 44, stratify=y)

###### Feature Scaling ###########################
# create object of class
sc = StandardScaler()

# Ensure to scale the filtered data as well
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

###### feature reduction ################
########## PCA ##########################
from sklearn.decomposition import PCA

# Perform PCA
pca = PCA(n_components=13)  # Choose the number of components
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Training data")
print(X_train_pca)

print("Testing data")
print(X_test_pca)

# %% ######################
# Train the Model
###########################
svm = train_model(X_train_pca, y_train)

# ##############################
# Predict test results
###################################
y_pred = svm.predict(X_test_pca)

# Convert y_test to numpy array
y_test_np = y_test.to_numpy()

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test_np.reshape(len(y_test),1)),1))

# Metrics
##### Imports ##########################
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pylab as plt

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

report = classification_report(y_test, y_pred, target_names=['No Whale Call', 'Whale call'])
print('Classification Report:')
print(report)


# %% ##############################
# Detect Table
##################################
detect_table_path = f'D:\\Universiteit\\5de jaar\\Skripsie Data\\DTW\\Test feature tables\\MFCC\\MFCCDetectFeatures_table_Test_4_2048_1024.csv'
detect_df = pd.read_csv(detect_table_path)
print(detect_df.head())

# #############################
# Predict segments
##################################\
#Scale the data
detect_df_scaled = sc.transform(detect_df)

# Feature reduction
# Perform PCA
detect_df_pca = pca.transform(detect_df_scaled)

#predict if call present in chunks
predictions = svm.predict(detect_df_pca)
print(predictions.reshape(len(predictions),1))

#%% ############################
# Merge and Write to text file
################################
textfile_path = 'D:\\Universiteit\\5de jaar\\Skripsie Data\\ST-42 Textfiles\\MFCC\\Call_times_Test_4_model9.txt'
write_to_textfile(predictions, textfile_path)


# %% ########################
# Metrices
#############################
from SVM_functions import confusion_matrix
#table paths
ground_truth_path = f'D:\\Universiteit\\5de jaar\\Skripsie Data\\ST-42 Testing Selection tables\\selection_table_Test_4_3.txt'
predicted_path = f'D:\\Universiteit\\5de jaar\\Skripsie Data\\ST-42 Textfiles\\MFCC\\Call_times_Test_4_model9.txt'

TP, FP, FN = confusion_matrix(ground_truth_path, predicted_path)

# Calculate Precision
if (TP + FP) > 0:
    precision = TP / (TP + FP)
else:
    precision = 0  # Avoid division by zero

# Calculate Recall
if (TP + FN) > 0:
    recall = TP / (TP + FN)
else:
    recall = 0  # Avoid division by zero
    
# Calculate F1 score
if (precision + recall) > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0
    
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")




