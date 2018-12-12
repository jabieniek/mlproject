import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt
import mappers as mp
import classifiersregressors as f
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#Create your df here:
df = pd.read_csv("profiles.csv")
#print(df.loc[0:5,["essay1","education","age","essay5"]])

df = df[df.age.notnull()]

df["agegroup"] = df.age.map(mp.ageMapper)
df["drinkcode"] = df.drinks.map(mp.drinksMapper)
df["smokecode"] = df.smokes.map(mp.smokesMapper)
df["bodycode"] = df.body_type.map(mp.bodyTypeMapper)
df["drugcode"] = df.drugs.map(mp.drugsMapper)

#print(df[["drinkcode","smokecode"]])

age = df[["agegroup"]]
features = df[["drinkcode","smokecode","bodycode","drugcode","income"]]

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(features)
df_normalized = pd.DataFrame(np_scaled,columns=["drinkcode","smokecode","bodycode","drugcode","income"])

x_train, x_test, y_train, y_test = train_test_split(features, age, train_size = 0.7, test_size = 0.3, random_state=6)

f.RunMLR(x_train,y_train,x_test,y_test)
f.RunKNNRegression(x_train,y_train,x_test,y_test)
#f.ChartKNNRegressor(x_train,y_train,x_test,y_test)
#f.ChartKNNClassifier(x_train,y_train,x_test,y_test)
f.RunKNNClassifier(x_train,y_train,x_test,y_test)
f.RunSVC(x_train,y_train,x_test,y_test)


