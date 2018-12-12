import numpy as np
import math
import pandas as pd

def ageMapper(age):
    if 18 <= age <= 20: return 1
    elif (21 <= age <=25): return 2
    elif (26 <= age <=30): return 3
    elif (31 <= age <=35): return 4
    elif (36 <= age <=40): return 5
    elif (41 <= age <=45): return 6
    elif (46 <= age <=50): return 7
    elif (51 <= age <=55): return 8
    elif (56 <= age <=60): return 9
    elif (61 <= age <=65): return 10
    elif (66 <= age <=70): return 11
    elif (71 <= age <=75): return 12
    elif (76 <= age <=80): return 13
    elif (81 <= age <=85): return 14
    elif (86 <= age <=90): return 15
    elif (91 <= age <=95): return 16
    elif (96 <= age <=100): return 17
    elif (101 <= age <=105): return 18
    elif (106 <= age <=110): return 19

#if they smoke at all it's a yes.
def smokesMapper(smokeVal):
    if(pd.isnull(smokeVal)): return -1
    if(smokeVal != 'no'):
        return 1
    else:
        return 0

def drugsMapper(drugVal):
    if(pd.isnull(drugVal)): return -1
    drugMap = {"never":0,"sometimes":1,"often":2}
    return drugMap[drugVal]

def drinksMapper(drinkVal):
    if(pd.isnull(drinkVal)): return -1
    drinkmaps = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
    return drinkmaps[drinkVal]
    
def bodyTypeMapper(bodyTypeVal):
    if(pd.isnull(bodyTypeVal)): return -1
    bodyTypeMap = {"average":0, 
    "fit":1,
    "athletic":2,
    "thin":3,
    "curvy":4,
    "a little extra":5,
    "skinny":6,
    "full figured":7,
    "overweight":8,
    "jacked":9,
    "used up":10,
    "rather not say":11}
    return bodyTypeMap[bodyTypeVal]