# Skripsie
This project focuses on detecting Bryde's whale vocalisations using machine learning techniques, specifically Support Vector Machined (SVM) and Dynamic Time Warping (DTW). The project includes features extraction, SVM classification, and DTW template matching, with modular functions for both SVM and DTW implementations. 

## Project Structure
- `SVM.py`
  
  This file contains the main code for implementing the SVM classifier. It uses extracted audio features to classify segments as either containing a whale call or not. The classifier is optimized for distinguishing whale calls based on pre-extracted features.

- `DTW.py`

  This file includes the DTW algorithm for aligning audio templates to identify patterns associated with whale calls. DTW allows for flexible matching, accommodating time variability in the whale sounds.

- `SVM_functions.py`

  This file contains helper functions specific to the SVM implementation, including functions for data preprocessing, training the SVM model, and evaluating classification results. It modularises tasks to keep the main SVM implementation file concise and organised.

- `DTW_functions.py`

  Similar to the `SVM_functions` file, this file provides reusable functions for the DTW implementation. These include functions for calculating DTW distances, aligning audio segments, and managing template matching tasks.

- `Extract_features.py`

  This file is dedicated to extracting relevant audio features (such as MFCCs) from raw audio data. It preprocesses audio segments and outputs feature vectors that are subsequently used by both the SVM and DTW algorithms.

## Getting Started

### Prerequisites

Make sure you have the following Python libraries installed:
- `pandas`
- `numpy`
- `librosa`
- `scikit-learn`
- `pydub`

Install them via:
```bash
pip install pandas numpy librosa scikit-learn pydub
```

### Usage
1. **Extract features:**
   
    Run the `Extract_features.py` file to process the audio data and generate feature vectors. Ensure the file paths are correct to where the audio data is stored on the device.

2. **Train SVM:**

   Use the `SVM.py` file to train the SVM classifier on labelled data, import new data, and detect the vocalisations. It will generate a textfile with the detected vocalisations that can be directly imported into Raven Pro as a selection table.

3. **DTW Matching:**

   Run the `DTW.py` file to perform template matching using DTW. This will also generate a textfile with the detected vocalisations to be imported into Raven Pro as a selection table.

5. **Helper Functions:**

   The `SVM_functions.py` and `DTW_functions.py` files is already imported within the SVM and DTW implementation files.

### Running Code Blocks with '#!' comments

In this project, you will notice the use of `#!` or "magic comments" at the beginning of some code blocks in the Python file. This allows you to run each code block individually and see the output immediately. By pressing **Run** at the top of each cell, you can execute the code, making it easy to test specific functions and view the outputs in real-time. 
