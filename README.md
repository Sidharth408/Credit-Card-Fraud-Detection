Credit Card Fraud Detection using Machine Learning
This project demonstrates the use of various machine learning models to detect fraudulent credit card transactions. The dataset used for this project is a publicly available credit card transaction dataset, which includes both fraudulent and non-fraudulent transactions. The goal is to build models that can accurately classify these transactions.

Table of Contents
Installation
Project Overview
Dataset
Modeling
Results
Model Comparison
License
Installation
To run this project, you'll need to install the following Python libraries:

bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
Project Overview
This project uses four different machine learning models to predict fraudulent transactions in credit card data:

Random Forest Classifier
Gradient Boosting Classifier
Support Vector Machine (SVM)
XGBoost Classifier
The models are evaluated based on their accuracy and performance metrics, including classification reports and confusion matrices.

Dataset
The dataset used in this project is from a public dataset of credit card transactions(download dataset from kaggle). The data contains several features, including:

Time: Time elapsed since the first transaction in the dataset
V1 to V28: Anonymized features representing different attributes of the transaction
Amount: The monetary amount of the transaction
Class: The target variable indicating whether the transaction is fraudulent (1) or not (0)
For more information about the dataset, you can refer to the source.

source : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Modeling
Data Preprocessing:

The dataset is loaded into a pandas DataFrame.
The Class column is separated as the target variable, and the other columns are used as features.
The RobustScaler is applied to the feature columns (excluding Time) to handle outliers more effectively.
The dataset is split into training and testing sets with an 80-20 split.
Model Training and Evaluation:

Four different machine learning classifiers are used:
Random Forest Classifier
Gradient Boosting Classifier
Support Vector Machine (SVM)
XGBoost Classifier
Each model is trained on the training set and evaluated on the test set using metrics such as accuracy, classification report, and confusion matrix.
Model Comparison:

A bar plot is generated to compare the accuracy scores of all models.
Results
The results from the models are shown in terms of accuracy, precision, recall, F1-score, and confusion matrices. These results are visualized using heatmaps for easy comparison and analysis.

Model Comparison
The following bar plot compares the accuracy of the four models:


The models are evaluated based on their classification performance, and the one with the highest accuracy will be considered for further refinement.
