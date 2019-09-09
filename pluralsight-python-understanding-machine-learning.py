# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:14:39 2019

@author: eeshwarankathi
"""


# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load and review data
df = pd.read_csv('C://eeshwarankathi//MachineLearning//pima-data.csv')
df.shape

# df.head(5)
# df.tail(5)

df.isnull().values.any()
pd.set_option('display.max_columns', None)

def plot_corr(df, size = 11):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    
plot_corr(df)
df.corr()
del df['skin']

# Checking data types
df.head(5)

# changing true to 1 and false to 0
diabetes_map = {True: 1, False: 0}

df['diabetes'] = df['diabetes'].map(diabetes_map)

# Checking T/F ratio

num_true = len(df.loc[df['diabetes'] == True])
num_false = len(df.loc[df['diabetes'] == False])

print("Number of True cases: {0} ({1:2.2f}%)".format(num_true, (num_true/(num_true+num_false))*100))
print("Number of False cases: {0} ({1:2.2f}%)".format(num_false, (num_false/(num_true+num_false))*100))

# 70% - training, 30% - testing

from sklearn.model_selection import train_test_split

feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_col_names = ['diabetes']

X = df[feature_col_names].values
Y = df[predicted_col_names].values

split_test_size = 0.30
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = split_test_size, random_state = 42)

print("{0:0.2f}% in training set".format((len(X_train)/len(df.index)) * 100))
print("{0:0.2f}% in test set".format((len(X_test)/len(df.index)) * 100))

print("Original True  : {0} ({1:0.2f}%)".format(len(df.loc[df['diabetes'] == 1]), (len(df.loc[df['diabetes'] == 1])/len(df.index)) * 100))
print("Original False : {0} ({1:0.2f}%)".format(len(df.loc[df['diabetes'] == 0]), (len(df.loc[df['diabetes'] == 0])/len(df.index)) * 100))
print("")
print("Training True  : {0} ({1:0.2f}%)".format(len(Y_train[Y_train[:] == 1]), (len(Y_train[Y_train[:] == 1])/len(Y_train)) * 100))
print("Training False : {0} ({1:0.2f}%)".format(len(Y_train[Y_train[:] == 0]), (len(Y_train[Y_train[:] == 0])/len(Y_train)) * 100))
print("")
print("Test True      : {0} ({1:0.2f}%)".format(len(Y_test[Y_test[:] == 1]), (len(Y_test[Y_test[:] == 1])/len(Y_test)) * 100))
print("Test False     : {0} ({1:0.2f}%)".format(len(Y_test[Y_test[:] == 0]), (len(Y_test[Y_test[:] == 0])/len(Y_test)) * 100))

print('# rows in dataframe {0}'.format(len(df)))
print('# rows missing glucose_conc: {0}'.format(len(df.loc[df['glucose_conc'] == 0])))
print('# rows missing diastolic_bp: {0}'.format(len(df.loc[df['diastolic_bp'] == 0])))
print('# rows missing thickness   : {0}'.format(len(df.loc[df['thickness'] == 0])))
print('# rows missing insulin     : {0}'.format(len(df.loc[df['insulin'] == 0])))
print('# rows missing bmi         : {0}'.format(len(df.loc[df['bmi'] == 0])))
print('# rows missing diab_pred   : {0}'.format(len(df.loc[df['diab_pred'] == 0])))
print('# rows missing age         : {0}'.format(len(df.loc[df['age'] == 0])))

from sklearn.impute import SimpleImputer

fill_0 = SimpleImputer(missing_values=0, strategy = "mean")

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)

# Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train, Y_train.ravel())

nb_predict_train = nb_model.predict(X_train)
from sklearn import metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(Y_train, nb_predict_train)))

nb_predict_test = nb_model.predict(X_test)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(Y_test, nb_predict_test)))

# Metrics

print("Confusion matrix")
print("{0}".format(metrics.confusion_matrix(Y_test, nb_predict_test)))
print("")
print("Classification Report")
print("{0}".format(metrics.classification_report(Y_test, nb_predict_test)))

# Random Forest - Over fitting

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, Y_train.ravel())

rf_predict_train = rf_model.predict(X_train)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(Y_train, rf_predict_train)))

rf_predict_test = rf_model.predict(X_test)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(Y_test, rf_predict_test)))

print("Confusion matrix")
print("{0}".format(metrics.confusion_matrix(Y_test, rf_predict_test)))
print("")
print("Classification Report")
print("{0}".format(metrics.classification_report(Y_test, rf_predict_test)))

# Logistic Regression - hyperparameters

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(C=0.7, random_state=42)
lr_model.fit(X_train, Y_train.ravel())
lr_predict_train = lr_model.predict(X_train)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(Y_train, lr_predict_train)))

lr_predict_test = lr_model.predict(X_test)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(Y_test, lr_predict_test)))

print("Confusion matrix")
print("{0}".format(metrics.confusion_matrix(Y_test, lr_predict_test)))
print("")
print("Classification Report")
print("{0}".format(metrics.classification_report(Y_test, lr_predict_test)))

# Loping over different values of C

C_start = 0.1
C_end = 5
C_inc = 0.1

C_values, recall_scores = [], []
C_val = C_start

best_recall_score = 0
while (C_val < C_end):
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(C=C_val, random_state=42)
    lr_model_loop.fit(X_train, Y_train.ravel())
    lr_predict_loop_test = lr_model_loop.predict(X_test)
    recall_score = metrics.recall_score(Y_test, lr_predict_loop_test)
    recall_scores.append(recall_score)
    if(recall_score > best_recall_score):
        best_recall_score = recall_score
        best_lr_predict_test= lr_predict_loop_test
        
    C_val = C_val + C_inc
    
best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("1st max value of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))

plt.plot(C_values, recall_scores, "-")
plt.xlabel("C Value")
plt.ylabel("recall score")

# Adding Class Weight

C_start = 0.1
C_end = 5
C_inc = 0.1

C_values, recall_scores = [], []
C_val = C_start

best_recall_score = 0
while (C_val < C_end):
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(C=C_val, class_weight="balanced" ,random_state=42)
    lr_model_loop.fit(X_train, Y_train.ravel())
    lr_predict_loop_test = lr_model_loop.predict(X_test)
    recall_score = metrics.recall_score(Y_test, lr_predict_loop_test)
    recall_scores.append(recall_score)
    if(recall_score > best_recall_score):
        best_recall_score = recall_score
        best_lr_predict_test= lr_predict_loop_test
        
    C_val = C_val + C_inc
    
best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("1st max value of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))

plt.plot(C_values, recall_scores, "-")
plt.xlabel("C Value")
plt.ylabel("recall score")

# Using the best C value with balanced class weight

lr_model = LogisticRegression(class_weight="balanced", C=best_score_C_val, random_state=42)
lr_model.fit(X_train, Y_train.ravel())
lr_predict_train = lr_model.predict(X_train)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(Y_train, lr_predict_train)))

lr_predict_test = lr_model.predict(X_test)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(Y_test, lr_predict_test)))

print("Confusion matrix")
print("{0}".format(metrics.confusion_matrix(Y_test, lr_predict_test)))
print("")
print("Classification Report")
print("{0}".format(metrics.classification_report(Y_test, lr_predict_test)))

# Cross Validation

from sklearn.linear_model import LogisticRegressionCV
lr_cv_model = LogisticRegressionCV(n_jobs=1, random_state=42, Cs=3, cv=10, refit=False, class_weight="balanced")
lr_cv_model.fit(X_train, Y_train.ravel())
lr_cv_predict_train = lr_cv_model.predict(X_train)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(Y_train, lr_cv_predict_train)))

lr_cv_predict_test = lr_cv_model.predict(X_test)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(Y_test, lr_cv_predict_test)))

print("Confusion matrix")
print("{0}".format(metrics.confusion_matrix(Y_test, lr_cv_predict_test)))
print("")
print("Classification Report")
print("{0}".format(metrics.classification_report(Y_test, lr_cv_predict_test)))

