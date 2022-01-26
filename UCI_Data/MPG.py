# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 15:12:40 2022

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import re
import os

# Windows File Path
def pathmaker(filename):
    file_name = filename
    fullfilename = os.path.join(os.path.dirname(__file__), file_name)
    return(fullfilename)

# Inputs
data_file_mpg = pathmaker('auto-mpg.csv')
fraction_of_training_data_required = 0.8

# Load Data
df_mpg = pd.read_csv(data_file_mpg, header=None, delim_whitespace=True)
#print(df_mpg) # Data Sanity Check

# Data Preparation
attribute_info ='''
    1. mpg:           continuous
    2. cylinders:     multi-valued discrete
    3. displacement:  continuous
    4. horsepower:    continuous
    5. weight:        continuous
    6. acceleration:  continuous
    7. model year:    multi-valued discrete
    8. origin:        multi-valued discrete
    9. car name:      string (unique for each instance)
'''
column_names_mpg = re.findall('(?<=\d\.\s)([a-z,\s]*)(?=\:)', attribute_info)
#print(column_names_mpg)
df_mpg = df_mpg.set_axis(column_names_mpg, axis=1, inplace=False)
#print(df_mpg.dtypes)
df_mpg["horsepower"] = df_mpg["horsepower"].replace(to_replace='[^\d.]', value=np.nan, regex=True) # convert non-numbers to NaN
df_mpg["horsepower"] = df_mpg["horsepower"].astype('float64')
#print(df_mpg.dtypes)

feature_list_mpg = column_names_mpg[:-1]
output_mpg = column_names_mpg[-1]

# Final Exploration
# print(df_mpg[feature_list_mpg].describe())
# print(df_mpg[output_mpg].describe())
# print(pd.crosstab(index=df_mpg[output_mpg], columns="count"))

# Data Visualization
fig, axes = plt.subplots(nrows=1, ncols=len(feature_list_mpg))
for feature in range(len(feature_list_mpg)):
    df_mpg.boxplot(column=feature_list_mpg[feature],ax=axes[feature])
plt.show()

# Train/Test Split
X = df_mpg[feature_list_mpg]
y = df_mpg[output_mpg]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-fraction_of_training_data_required, random_state = 0)

# Modelling ************************************
# Feature Scaling
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
print(len(X_train_sc[0,:]), len(X_train_sc[:,0]))

# Running the model
def classifier(model, X_train, X_test, y_train, y_test):
    # failsafe for NaN --- Treat as mean
    for x in [X_train, X_test]:
        for col in range(len(x[0,:])):
            if np.isnan(x[:,col]).any():
                x[:,col] = np.nan_to_num(x[:,col], nan=np.nanmean(x[:,col]))
    for y in [X_train, X_test]:
        if np.isnan(y).any():
            y = np.nan_to_num(y, nan=np.nanmean(y))

    model.fit(X_train, y_train) # Train
    y_pred = model.predict(X_test) # Apply Model
    
    cm = confusion_matrix(y_test, y_pred)
    ac = accuracy_score(y_test,y_pred)
    
    return cm, ac

# Different Classifiers
NB_cm, NB_ac = classifier(GaussianNB(), X_train_sc, X_test_sc, y_train, y_test)
kNN_cm, kNN_ac = classifier(KNeighborsClassifier(n_neighbors=13), X_train_sc, X_test_sc, y_train, y_test)
SVM_cm, SVM_ac = classifier(svm.SVC(), X_train_sc, X_test_sc, y_train, y_test)

print(NB_cm)
print(NB_ac)
print(kNN_cm)
print(kNN_ac)
print(SVM_cm)
print(SVM_ac)

# Finding the best parameter for kNN
k_list = np.arange(3, 21, 1)
acc_score = []
for k in k_list:
    kNN_new_cm, kNN_new_ac = classifier(KNeighborsClassifier(n_neighbors=k), X_train_sc, X_test_sc, y_train, y_test)
    acc_score.append(kNN_new_ac)

plt.plot(k_list, acc_score)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

df_classifiers = pd.DataFrame([])
df_classifiers["Classifier_Descriptions"] = ["Naive-Bayes", "k Nearest Neighbours", "Support Vector Machine"]
df_classifiers["parameters"] = ["", "k=13", ""]
df_classifiers["accuracy_score"] = [NB_ac, kNN_ac, SVM_ac]

print(df_classifiers)
