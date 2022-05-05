---
title: Identification
---

### Files:
* "CNN-LSTM.py" is a CNN+LSTM method used for identification
* "CNN.py" is a CNN method used for identification
* "lstm_fix.py" is a CNN+LSTMfix method used for identification

### How to use the codes:
* Step 1: Enter your training dataset path on line 131 (*Note: Folder name will be train/inertial signals)
* Step 2: Enter your testing dataset path on line 134 (*Note: Folder name will be test/inertial signals)
* Step 3: Enter your training label path on line 137 (*Note: train/y_train.txt is the label file)
* Step 4: Enter your testing label path on line 140 (*Note: test/y_train.txt is the label file)
* Step 5: Run the code to train and save model
* Step 6: Use the saved model to identify gaits from your dataset using only walking data
