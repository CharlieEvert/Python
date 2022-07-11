#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 15:28:34 2022

@author: charlieevert
"""
#data from https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay

#This project aims to predict flight delays from a Kaggle dataset using a few Machine Learning Classification Models
import pandas as pd

#read the csv, remove id and flight no columns
df_main = pd.read_csv (r'/Users/charlieevert/Downloads/Airlines.csv', header='infer')
df_main = df_main.drop(columns=['id','Flight','AirportTo','AirportFrom','Length'])

#filter it to a random sample of 10k rows
df = df_main.sample(n=10000, random_state=1)
df.head()

#change date to categorical variable
df['DayOfWeek'] = df['DayOfWeek'].replace([1],'Sunday')
df['DayOfWeek'] = df['DayOfWeek'].replace([2],'Monday')
df['DayOfWeek'] = df['DayOfWeek'].replace([3],'Tuesday')
df['DayOfWeek'] = df['DayOfWeek'].replace([4],'Wendesday')
df['DayOfWeek'] = df['DayOfWeek'].replace([5],'Thursday')
df['DayOfWeek'] = df['DayOfWeek'].replace([6],'Friday')
df['DayOfWeek'] = df['DayOfWeek'].replace([7],'Saturday')

#dummy code weekday and airline
df = pd.concat([df.drop('DayOfWeek', axis=1), pd.get_dummies(df['DayOfWeek'])], axis=1)
df = pd.concat([df.drop('Airline', axis=1), pd.get_dummies(df['Airline'],prefix='Airline_')], axis=1)

df.info()

#split into test/train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df,df['Delay'],test_size=0.2,random_state=0)

#grid search
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
test_score = grid_search.score(X_test, y_test)
print("Test set score {:.2f}".format(test_score))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
print("Best estimator:\n{}".format(grid_search.best_estimator_))

#create models
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
pipeline_lr=Pipeline([('scalar1',StandardScaler()),
                     ('pca1',PCA(n_components=2)), 
                     ('lr_classifier',LogisticRegression())])
pipeline_dt=Pipeline([('scalar2',StandardScaler()),
                     ('pca2',PCA(n_components=2)),
                     ('dt_classifier',DecisionTreeClassifier())])
pipeline_svm=Pipeline([('scalar3', StandardScaler()),
                      ('pca3', PCA(n_components=2)),
                      ('clf', svm.SVC())])
pipeline_knn=Pipeline([('scalar4',StandardScaler()),
                     ('pca4',PCA(n_components=2)),
                     ('knn_classifier',KNeighborsClassifier())])
pipelines = [pipeline_lr, pipeline_dt, pipeline_svm, pipeline_knn]
pipe_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'Support Vector Machine',3:'K Nearest Neighbor'}
    
#create y predictions and fits for each model
for pipe in pipelines:
  pipe.fit(X_train, y_train)

#test accuracy of models
for i,model in enumerate(pipelines):
    print("{} Test Accuracy:{}".format(pipe_dict[i],model.score(X_test,y_test)))

#K-Fold cross validation
accuracies_lr = cross_val_score(estimator = pipeline_lr, X = X_train, y = y_train, cv = 10)
print(accuracies_lr.mean())
print(accuracies_lr.std())

accuracies_dt = cross_val_score(estimator = pipeline_dt, X = X_train, y = y_train, cv = 10)
print(accuracies_dt.mean())
print(accuracies_dt.std())

accuracies_svm = cross_val_score(estimator = pipeline_svm, X = X_train, y = y_train, cv = 10)
print(accuracies_svm.mean())
print(accuracies_svm.std())

accuracies_knn = cross_val_score(estimator = pipeline_knn, X = X_train, y = y_train, cv = 10)
print(accuracies_knn.mean())
print(accuracies_knn.std())

#creating y predictions from each model
#knn
y_preds_knn = pipeline_knn.predict(X_test)
y_preds_lr = pipeline_lr.predict(X_test)  
y_preds_svm = pipeline_svm.predict(X_test)
y_preds_dt = pipeline_dt.predict(X_test) 

#confusion matrices
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
cmKNN = confusion_matrix(y_test, y_preds_knn)
cmLR = confusion_matrix(y_test, y_preds_lr)
cmSVM = confusion_matrix(y_test, y_preds_svm)
cmDT = confusion_matrix(y_test, y_preds_dt)

#visualize each
sns.heatmap(cmKNN, annot=True,cmap="Blues",xticklabels='auto', yticklabels='auto', fmt='g').set(title='KNN')
plt.xlabel('Test Results', fontsize = 15) 
plt.ylabel('Predicted Results', fontsize = 15) 
sns.heatmap(cmLR, annot=True,cmap="Greens",xticklabels='auto', yticklabels='auto', fmt='g').set(title='LR')
plt.xlabel('Test Results', fontsize = 15) 
plt.ylabel('Predicted Results', fontsize = 15) 
sns.heatmap(cmSVM, annot=True,cmap="Purples",xticklabels='auto', yticklabels='auto', fmt='g').set(title='SVM')
plt.xlabel('Test Results', fontsize = 15) 
plt.ylabel('Predicted Results', fontsize = 15) 
sns.heatmap(cmDT, annot=True,cmap="Reds",xticklabels='auto', yticklabels='auto', fmt='g').set(title='DT')
plt.xlabel('Test Results', fontsize = 15) 
plt.ylabel('Predicted Results', fontsize = 15) 

#classification reports
from sklearn.metrics import classification_report
target_names = ['No Delay', 'Delay']
#knn
print(classification_report(y_test, y_preds_knn, target_names=target_names))
#LR
print(classification_report(y_test, y_preds_lr, target_names=target_names))
#SVM
print(classification_report(y_test, y_preds_svm, target_names=target_names))
#DT
print(classification_report(y_test, y_preds_dt, target_names=target_names))
