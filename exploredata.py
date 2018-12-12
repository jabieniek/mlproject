import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt
import mappers as mp




#Create your df here:
df = pd.read_csv("profiles.csv")
#print(df.loc[0:5,["essay1","education","age","essay5"]])

data = df[["age","income","smokes","drinks","drugs","body_type"]]

df["agegroup"] = df.age.map(mp.ageMapper)
df["drinkcode"] = df.drinks.map(mp.drinksMapper, na_action='ignore')
df["smokecode"] = df.smokes.map(mp.smokesMapper, na_action='ignore')
df["bodycode"] = df.body_type.map(mp.bodyTypeMapper, na_action='ignore')
df["drugcode"] = df.drugs.map(mp.drugsMapper, na_action='ignore')


# plt.scatter(df.agegroup,df.income,alpha=.3)
# plt.xlabel("Age Grouping")
# plt.ylabel("Income")
# plt.show()

plt.scatter(df.agegroup,df.drinkcode,alpha=.3)
plt.xlabel("Age Grouping")
plt.ylabel("Drink Code")
plt.yticks(np.arange(6),labels=("not at all","rarely","socially","often","very often","desperately"))
plt.show()

# plt.scatter(df.agegroup,df.bodycode,alpha=.3)
# plt.xlabel("Age Grouping")
# plt.ylabel("Body Code")
# plt.show()




