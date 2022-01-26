# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 13:00:46 2022

@author: Shubham
"""
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score

# Inputs
data_file = 'Social_Network_Ads.csv'
fraction_of_training_data_required = 0.8

# Load Data
df_data = pd.read_csv(data_file)
print(df_data) # Data Sanity Check

# Initial Exploration **
list_data_columns = df_data.columns
print(list_data_columns)
print(df_data.dtypes)

''' ****************************************
Features: 'Gender', 'Age', 'EstimatedSalary'
Output: 'Purchased'
*****************************************'''

# Data Preparation
df_data['Gender'] = df_data['Gender'].astype('category')
print(df_data.dtypes)
df_data['Gender_Category'] = df_data['Gender'].cat.codes # Encoding
print(dict(enumerate(df_data['Gender'].cat.categories))) # Category Index List

feature_list = ['Gender_Category', 'Age', 'EstimatedSalary']
feature_list_revised = ['Age', 'EstimatedSalary']
output_column = 'Purchased'

# Final Exploration
print(df_data[feature_list].describe())

# Data Visualization
fig, axes = plt.subplots(nrows=1, ncols=2)
df_data.boxplot(column='Age',ax=axes[1])
df_data.boxplot(column='EstimatedSalary', ax=axes[0])

# Train/Test Split
X = df_data[feature_list]
y = df_data[output_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-fraction_of_training_data_required, random_state = 0)

# Modelling ************************************
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

NB_classifier = GaussianNB() # Naive-Bayes Classifier
NB_classifier.fit(X_train, y_train) # Train
NB_y_pred = NB_classifier.predict(X_test) # Apply Model

NB_cm = confusion_matrix(y_test, NB_y_pred)
NB_ac = accuracy_score(y_test,NB_y_pred)

print(NB_cm)
print(NB_ac)

# Conclusion for NB
'''******************************************
1.Scaling doesn't affect accuracy
2.Gender Category doesn't affect accuracy
******************************************'''