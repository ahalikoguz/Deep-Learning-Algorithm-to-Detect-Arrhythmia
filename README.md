# Deep-Learning-Algorithm-to-Detect-Arrhythmia
This repository contains a Python script for classifying electrocardiogram (ECG) data using a deep learning model. The script is designed to process ECG records, predict cardiac conditions, and evaluate the performance of the model. Key features include:

    Data Processing: Loads and preprocesses ECG data from text files, ensuring they match the expected format (5000 samples, 12 channels after removing the first non-feature column).
    
    Model Loading and Inference: Utilizes a pre-trained deep learning model (specifically, a version of the RegNetY-800MF model) for ECG classification.
    The model is loaded using PyTorch, and its performance is evaluated on the processed data.
    
    Prediction and Probability Calculation: For each ECG record, the script predicts the cardiac condition (AFIB, Flutter, or Sinus) and calculates the corresponding probability of the prediction.
    
    Result Display: Outputs the predicted class and its probability for each sample in the ECG data file.

Usage

Ensure you have Python and PyTorch installed. Place your ECG data files in the specified format in the same directory as the script, and update the file_name variable in the script with your file name. Run the script to see the classification results for each ECG sample in the file.
Requirements

    Python
    NumPy
    PyTorch

This script is part of our research work on applying deep learning techniques for accurate and efficient ECG analysis. For more information on our methodology and results, refer to our accompanying research article.
