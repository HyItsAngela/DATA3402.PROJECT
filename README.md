![](UTA-DataScience-Logo.png)

# Project Title

* This repository holds an attempt to apply machine learning techniques and models to metastatic cancer diagnosis to predict id patients recieved a cancer diagnosis withing 90 days of screening using data from
"WiDS Datathon 2024 Challenge #1" Kaggle challenge [(https://www.kaggle.com/competitions/widsdatathon2024-challenge1)]. 

## Overview

* The task, as defined by the Kaggle challenge is to develop a model to predict if patients recieved metatstatic cancer diagnosis within 90 days of screening. This repository approaches this problem as a binary classification task, using 2 different models, Random Forest Classifier and a gradient boost model called, CatBoost. CatBoost was the best model for the task as it was able to predict whether a patient was diagnosed metastatic cancer within 90 days of screening scored at ~81% accuracy. At the time of this writing, the best performance on the Kaggle leaderboards of this metric is 82%.

## Summary of Work Done

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
![Screenshot 2024-05-03 112608](https://github.com/HyItsAngela/DATA3402.PROJECT/assets/143844332/24c2d6eb-f3b3-4fa7-b1c2-6f965fe4acdf)
Zoomed in example distributions for a couple features.

![Screenshot 2024-05-03 113355](https://github.com/HyItsAngela/DATA3402.PROJECT/assets/143844332/05641d3c-308a-41a5-bda0-bbac01102c8d)
Example distributions of more features


### Problem Formulation

* Train information about demographics, diagnosis and treatment options, insurance and more with machine learning to provide a better view about aspects that may contribute to health equity.
  * Models
    * 2 models were used: RandomForest and Catboost. Both are known for their robustness with noisy or not preprocessed well data. CatBoost was the best model according to it's predictive power and accuracy.
  * No in-depth fine-tuning to the models such as hypyerparameters, feature importance or cross validation were done. 

### Training

* Describe the training:
  * Training was done on a Surface Pro 9 using Python on jupyter notebook.
  * Training did not take long to process, with the longest load time to be approximately 30 seconds.
  * Looking at the ROC AUC curves and measurements gave substanstial insight into which model was performing better. Other methods of evaluation was a classification report, log loss (a personal evaluation metric that is tuned by CatBoost), and an accuracy score that was imported from scikit-learn.
  * Training was stopped early because of time constraints but made sure to incorporate a working model with plenty of evaluation metrics for comparison.
  * Had difficulty with one-hot encoding the categorical columns as it looks like it encoded the numerical columns too. This dataset was used on the RandomForest model so it may skew some features.

### Performance Comparison

* Key performance metrics used were log loss ( a metric personally tuned from CatBoost), and metrics imported from scikit-learn, classification report, accuracy score, and ROC and AUC.
* Show one (or few) visualization(s) of results, for example ROC curves.
  
  ![Screenshot 2024-05-03 075010](https://github.com/HyItsAngela/DATA3402.PROJECT/assets/143844332/d818b562-9c5e-4ef7-9e97-b64c6525bfb1)
ROC curve and AUC measurement comparisons for models, RandomForest and CatBoost. The higher the AUC the better the model, CatBoost has a higher AUC (0.78).

### Conclusions

* CatBoost worked better than RandomForest, however two different preprocessed datasets were used so this may not be a fair comparison. CatBoost could have yeilded better accuracy if time was taken to perform hyper tuning, trianing with important features, and using cross validation techniques, but as seen from it's ~81% accuracy the model did quite well with it's default training techniques and shines through with it's robustness and preprocessing techniques it's known for.

### Future Work

* For future study, I would definitely like to dive deeper into feature importance and hyper tuning to see what aspects affect not only the model but the diagnosis and treatment of the patients to reall study the healthcare equity.
* Other studies can dive into the geographical and environmentel pollutants that patients are exposed to as these two group of features seemed to be more correlated to the training as one would think.

## How to reproduce results

* The notebooks are well organized but further explanation, during preprocessing CatBoost uses a dataset that imputed substantial missing values with the average of that same feature (BMI, patient race, payer type) in an age group, other missing values were under 2% of the data, hence the rows being dropped. Redundant or features with only one unique value were dropped as they either crowded the data or didn't provide enough variability to be of proper use. RandomForest used a dataset that was processed even further than the CatBoost dataset as the data has been normalized and the categorical columns have been one-hot encoded and the original categorical columns have been dropped.
* As long as a platform that can provide Python code is used such as Collab, Anaconda, etc, is used, results can be replicated.

### Overview of files in repository

* The repository includes 5 files in total.
  * Initial Look.py: Introduces the analyst to the data by looking and understanding the features, missing values, and class imbalances
  * Data Preparation.ipynb: Applies the understaning from Intial Look.ipynb and begins to understand correlations and start cleaning, preprocessing, and visualizing.
  * Machine.ipynb: Trains, predicts, evaluates, and visualizes the data using machine learning models.
  * Kaggle Tabular Data.ipynb: Contains the outline for the project.
  * submission.csv: submission file for the Kaggle challenge that includes the final predictions of the model.

### Software Setup
* Make sure to have pandas, matplotlib, math, scipy.stats, catboost, and scikit-learn packages such as, .preprocessing, .model_selection, .ensemble, and .metrics installed and ready for use.

### Data

* Data can be downloaded through the official Kaggle website through the link stated above. Or through Kaggle's API interface.

### Training

* Split the train dataset into a training and test dataset that is otherwise known as the validation set. Initialize the ML model by calling upon the algorithm. ex) "model = RandomForestClassifier()". Predict the validation set using prediction techniques from the ML package. 

#### Performance Evaluation

* Evaluation metrics are imported such as the log loss (from CatBoost package), accuracy score, classification score. The ROC curve and AUC measurement were also imported and then placed into a function for comparison of multiple models.


## Citations

* Provide any references.







