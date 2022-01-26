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

# Inputs
data_file_iris = 'iris.data'
fraction_of_training_data_required = 0.8

# Load Data
df_iris = pd.read_csv(data_file_iris, header=None)
print(df_iris) # Data Sanity Check

# Data Preparation
'''
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class: 
      -- Iris Setosa
      -- Iris Versicolour
      -- Iris Virginica
'''
column_names_iris = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'iris_class']
df_iris = df_iris.set_axis(column_names_iris, axis=1, inplace=False)
print(df_iris)

df_iris["iris_class"] = df_iris["iris_class"].replace(to_replace=r'Iris\-', value='', regex=True) # Remove redundant "Iris-"
print(df_iris)

feature_list_iris = column_names_iris[:-1]
output_iris = column_names_iris[-1]

# Final Exploration
print(df_iris[feature_list_iris].describe())
print(df_iris[output_iris].describe())
print(pd.crosstab(index=df_iris[output_iris], columns="count"))

# Data Visualization
fig, axes = plt.subplots(nrows=1, ncols=len(feature_list_iris))
for feature in range(len(feature_list_iris)):
    df_iris.boxplot(column=feature_list_iris[feature],ax=axes[feature])
plt.show()

# Train/Test Split
X = df_iris[feature_list_iris]
y = df_iris[output_iris]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-fraction_of_training_data_required, random_state = 0)

# Modelling ************************************
# Feature Scaling
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# Running the model
def classifier(model, X_train, X_test, y_train, y_test):
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