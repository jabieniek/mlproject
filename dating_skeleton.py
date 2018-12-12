import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt
import mappers as mp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

mlr = LinearRegression()
mlr.fit(x_train,y_train)
y_predict = mlr.predict(x_test)


#print(y_predict)
print("MLR: " + str(mlr.score(x_test,y_test)))
# plt.scatter(y_test,y_predict,alpha=.3)
# plt.xlim(0,19)
# plt.ylim(0,19)
# plt.show()

classifier = KNeighborsClassifier(n_neighbors = 6)
classifier.fit(x_train,y_train)
KAgeGuess = classifier.predict(x_test)
# plt.scatter(y_test,KAgeGuess,alpha=.3)
# plt.xlim(0,19)
# plt.ylim(0,19)
# plt.show()
print("K Score: " + str(classifier.score(x_test,y_test)))
print(accuracy_score(y_test,KAgeGuess))
print(recall_score(y_test,KAgeGuess,average='micro'))
print(precision_score(y_test,KAgeGuess,average='micro'))
print(f1_score(y_test,KAgeGuess,average='micro'))

#svc classifier
svcclassifier = SVC(gamma=5,C=.7)
svcclassifier.fit(x_train,y_train)
svcPredict = svcclassifier.predict(x_test)
print("svc Score: " + str(svcclassifier.score(x_test,y_test)))
print(accuracy_score(y_test,svcPredict))
print(recall_score(y_test,svcPredict,average='micro'))
print(precision_score(y_test,svcPredict,average='micro'))
print(f1_score(y_test,svcPredict,average='micro'))


