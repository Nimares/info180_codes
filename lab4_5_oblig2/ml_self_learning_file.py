#1. Load libraries
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from numpy import nan


#2. -----loading data-----
names = ['pregnancy_rate', 'glucose_concentration', 'blood_pressure', 'skin_thickness', 
         'insulin_levels','BMI', 'diabetes_inheretance_function', 'age', 'diabets_diagnosis' ]

dataset = read_csv('pima-indians-diabetes.csv', names=names)

#.----summarizing data-----

#3.1 Shape (instances(rows) and attributes(columns))
#print(dataset.shape)

#3.2 peeking at data
    #head = start from top of file, and how many rows
#print(dataset.head(20))

#3.3 statistical summary
#print(dataset.describe())

#3.4 class distribution
    #number of rows with a given value
#print(dataset.groupby('blood_pressure').size())


#*-*-*-*- DEALING WITH MISSING VALUES *-*-*-*-*-
#1. ----- finding missing values -----
#1.1 using statistical summary to look at an overview (some have val of 0)
#print(dataset.describe())

#1.2finding number of missing values for each column
# num_missing = (dataset[['glucose_concentration', 'blood_pressure', 'skin_thickness', 
#          'insulin_levels','BMI']] == 0).sum()

#print(num_missing)

#1.3.1 marking missing values as NaN
dataset[['glucose_concentration', 'blood_pressure', 'skin_thickness', 
         'insulin_levels','BMI']] = dataset[['glucose_concentration', 'blood_pressure', 'skin_thickness', 
         'insulin_levels','BMI']].replace(0, nan)

#1.3.2 overview of number of rows with nan values
# print(dataset.isnull().sum())

#1.3.2 confirming replacements (no 0's only nan shown)
# print(dataset.head(30))


#2. ----- removing rows with missing values -----
# print(dataset.shape)
dataset.dropna(inplace=True)
# print(dataset.shape)


#*-*-*-*- Evaluating models *-*-*-*-*-
#5.1 split out validation dataset

# returns numpy representation of the DF
array = dataset.to_numpy()

# # input features
x = array[:, 0:8]

# # target features
y = array[:,8]

# training and test datasets with input and target features. 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1, stratify=y)

# #min_max scaling 
# min_max_scaler = MinMaxScaler()
# x_train_min_max = min_max_scaler.fit_transform(x_train)
# x_test_min_max = min_max_scaler.fit_transform(x_test)


# #method for finding nearest odd number 
# def nearest_odd(n):
#     n = int(round(n))
#     return n - 1 if n % 2 == 0 else n

# #deciding k value based on square root of number of samples
# k = nearest_odd(np.sqrt(len(x_train_min_max)))


# #making predictions
# knn_model = KNeighborsClassifier(n_neighbors=k)
# knn_model.fit(x_train_min_max, y_train)
# knn_predictions = knn_model.predict(x_test_min_max)

# #evaluating predictions
# print(accuracy_score(y_test, knn_predictions))
# print(confusion_matrix(y_test, knn_predictions))
# print(classification_report(y_test, knn_predictions))

#--------------------------------------Logistic Regression ---------------------------------------------------

#Instantiate model (Check that they all have same random state)
logreg_model = LogisticRegression(random_state=1) 

#Fit the model with data
logreg_model.fit(x_train, y_train)

#making prediction
logreg_output_prediction = logreg_model.predict(x_test)

#Model evaluation
print(accuracy_score(y_test, logreg_output_prediction))
print(confusion_matrix(y_test, logreg_output_prediction))
print(classification_report(y_test, logreg_output_prediction))
#------------------------------------- Categorical naieve bayes  ---------------------------------------------
#Category labels
q_labels = ['low', 'low_med', 'high_med', 'high']


#Assigning categories based on quarterlies
dataset['pregnancy_rate']  = pd.qcut(dataset['pregnancy_rate'], q=4, labels=q_labels)
dataset['glucose_concentration']  = pd.qcut(dataset['glucose_concentration'], q=4, labels=q_labels)
dataset['blood_pressure']  = pd.qcut(dataset['blood_pressure'], q=4, labels=q_labels)
dataset['skin_thickness']  = pd.qcut(dataset['skin_thickness'], q=4, labels=q_labels)
dataset['insulin_levels']  = pd.qcut(dataset['insulin_levels'], q=4, labels=q_labels)
dataset['BMI']  = pd.qcut(dataset['BMI'], q=4, labels=q_labels)
dataset['diabetes_inheretance_function']  = pd.qcut(dataset['diabetes_inheretance_function'], q=4, labels=q_labels)
dataset['age']  = pd.qcut(dataset['age'], q=4, labels=q_labels)


array2=OrdinalEncoder().fit_transform(dataset)


# input features
x = array2[:, 0:8]

# target features
y = array2[:,8]

#training and test datasets with input and target features. 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1, stratify=y)

# #making predictions
cnb_model = CategoricalNB()
cnb_model.fit(x_train, y_train)
cnb_predictions = cnb_model.predict(x_test)

#evaluating predictions
print(accuracy_score(y_test, cnb_predictions))
print(confusion_matrix(y_test, cnb_predictions))
print(classification_report(y_test, cnb_predictions))