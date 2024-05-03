![](UTA-DataScience-Logo.png)

# Project Title

* This repository holds an attempt to apply machine learning techniques and models to metastatic cancer diagnosis to predict id patients recieved a cancer diagnosis withing 90 days of screening using data from
"WiDS Datathon 2024 Challenge #1" Kaggle challenge [(https://www.kaggle.com/competitions/widsdatathon2024-challenge1)]. 

## Overview

* The task, as defined by the Kaggle challenge is to develop a model to predict if patients recieved metatstatic cancer diagnosis within 90 days of screening. This repository approaches this problem as a binary classification task, using 2 different models, Random Forest Classifier and a gradient boost model called, CatBoost. CatBoost was the best model for the task as it was able to predict whether a patient was diagnosed metastatic cancer within 90 days of screening scored at ~81% accuracy. At the time of this writing, the best performance on the Kaggle leaderboards of this metric is 82%.

## Summary of Workdone

Include only the sections that are relevant an appropriate.

### Data

* Data:
  * Type: Binary Classification
    * Input: CSV file: train.csv, test.csv -> diagnosis
    * Input: CSV file of features, output: cancer/no cancer flag in 1st column.
  * Size: Original training and testing datasets together was 16 MB. After cleaning and proper preprocessing both datasets together was about 36 MB.
  * Instances (Train, Test, Validation Split): 12906 patients for training, 5792 for testing, none for validation

#### Preprocessing / Clean up

* Dropped features that only had one unique values as they lacked predictive power and variability, and dropped redundant features. During visualization and coreelation patient age seemed to be the most trusted feature so missing values were imputed by the average of the age group. The rest of the missing values in other features that were below 2%, the rows were dropped. One-hot encoding and normalization was used on the cleaned data,

#### Data Visualization

Show a few visualization of the data and say a few words about what you see.

### Problem Formulation

* Define:
  * Input / Output
  * Models
    * Describe the different models you tried and why.
  * Loss, Optimizer, other Hyperparameters.

### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.







