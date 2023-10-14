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

# 1. ----------------- Loading and preprocessing ------------------
# 1.1 loading data

# names for headers
names = ['pregnancy_rate', 'glucose_concentration', 'blood_pressure', 'skin_thickness', 
         'insulin_levels','BMI', 'diabetes_inheretance_function', 'age', 'diabets_diagnosis' ]

#load csv file with data
dataset = read_csv('pima-indians-diabetes.csv', names=names)

# 1.2 preprocessing.

# 1.2.1 finding missing values
# statistical summary to look at an overview (some have val of 0)
# print(dataset.describe())

# 1.2.2. finding number of missing values for each column
# num_missing = (dataset[['glucose_concentration', 'blood_pressure', 'skin_thickness', 
#          'insulin_levels','BMI']] == 0).sum()
# print(num_missing)

#1.2.3 marking missing values as nan
dataset[['glucose_concentration', 'blood_pressure', 'skin_thickness', 
         'insulin_levels','BMI']] = dataset[['glucose_concentration', 'blood_pressure', 'skin_thickness', 
         'insulin_levels','BMI']].replace(0, nan)

# 1.2.4 overview of number of rows with nan values
# print(dataset.isnull().sum())

# 1.2.5 confirming replacements (no 0's only nan shown)
# print(dataset.head(30))


# 1.2.6 removing rows with missing values
# print(dataset.shape)
dataset.dropna(inplace=True)
# print(dataset.shape)


# 2. ------------- Splitting data into test and training -------------------------
# returns numpy representation of the DF
array = dataset.to_numpy()

# input features [row_start:row_end, column_start:column_end]
x = array[:, 0:8]

# target features [row_start:row_end, column_start:column_end]
y = array[:,8]

# seperating files into train and test for both input and target features. 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1, stratify=y)


# 3. ------------------------ Scaling ------------------------------
# min_max scaling 
min_max_scaler = MinMaxScaler()
x_train_min_max = min_max_scaler.fit_transform(x_train)
x_test_min_max = min_max_scaler.fit_transform(x_test)


# function for finding nearest odd number
def nearest_odd(n):
    n = int(round(n))
    return n - 1 if n % 2 == 0 else n

# deciding k value based on square root of number of samples
# recommendation found at: https://discuss.analyticsvidhya.com/t/how-to-choose-the-value-of-k-in-knn-algorithm/2606/2
k = nearest_odd(np.sqrt(len(x_train_min_max)))

# if more time remaining, attempt vizualisation for optimal k value shown in link below
# https://datascience.stackexchange.com/questions/69415/k-nearest-neighbor-classifier-best-k-value


# 4. ------------------------ Algorithm: K-nearest neighbour ----------------------------------------------
# Instantiate model 
knn_model = KNeighborsClassifier(n_neighbors=k)

# Fit model 
knn_model.fit(x_train_min_max, y_train)

# Prediction
knn_predictions = knn_model.predict(x_test_min_max)

# Model evaluation
print(accuracy_score(y_test, knn_predictions))
print(confusion_matrix(y_test, knn_predictions))
print(classification_report(y_test, knn_predictions))

# 5. --------------------------------------Algorithm: Logistic Regression -----------------------------------

# Instantiate model (Check that they all have same random state)
logreg_model = LogisticRegression(random_state=1) 

# Fit the model with data
logreg_model.fit(x_train, y_train)

# Prediction
logreg_output_prediction = logreg_model.predict(x_test)

#Model evaluation
print(accuracy_score(y_test, logreg_output_prediction))
print(confusion_matrix(y_test, logreg_output_prediction))
print(classification_report(y_test, logreg_output_prediction))
# 6. ------------------------------------- Categorical naieve bayes  ---------------------------------------------

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

#Ordinal encoding 
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