import pandas as pd
import numpy as np
import glob
import warnings
warnings.simplefilter('ignore')

path = r'data'
csv_files = glob.glob(path + "/*.csv")
data_files = glob.glob(path + "/*.data")

li = {}

for filename in csv_files:
    print(filename)
    if filename[len(path)+1:-4] == "naval_propulsion":
        df = pd.read_csv(filename, header = None, sep = "  ")
    else :
        df = pd.read_csv(filename, header = None)
    li[filename[len(path)+1:-4]] = df
    
for filename in data_files:
    print(filename)
    df = pd.read_csv(filename, header = None)
    li[filename[len(path)+1:-5]] = df


def process(df, cy, cdrop = [], header = False, dummies = []):
    if header :
        df = df.drop(0, axis = 0)
    ndummies = []
    for d in dummies :
        ndrop = 1 if d > cy else 0
        for c in cdrop :
            if d > c :
                ndrop+= 1
        ndummies.append(d-ndrop)
    cols = [x for x in range(df.shape[1])]
    for c in cdrop:
        cols.remove(c)
    df1 = df.iloc[:, cols].dropna(axis = 0)
    ncy = cy
    for c in cdrop :
        if cy > c :
            ncy-=1
    Y = df1.iloc[:, ncy]
    cols1 = [x for x in range(df1.shape[1])]
    cols1.remove(ncy)
    df2 = df1.iloc[:, cols1]
    for d in ndummies : 
        dcod = pd.get_dummies(df2.iloc[:, d], drop_first = True)
        dcod = dcod.set_axis(["cat_"+str(d)+str(i) for i in dcod.columns], axis = 1)
        df2 = df2.join(dcod)
    cols = [x for x in range(df2.shape[1])]
    rmcols = ndummies
    for r in rmcols :
        cols.remove(r)
    X = df2.iloc[:, cols]
    X = X.astype(float)
    Y = Y.astype(float)
    return X, Y


dFrame = {"blog" : {"cy" : 280, "cdrop" : [277, 54, 262]},
          "fbmetrics" : {"cy" : 18, "cdrop" : [15, 16, 17], "header" : True, "dummies" : [1]},
          "fbcom1" : {"cy" : 53, "cdrop" : [33, 39, 46]},
          "fbcom2" : {"cy" : 53, "cdrop" : [33, 39, 46]},
          "fbcom3" : {"cy" : 53, "cdrop" : [33, 39, 46]},
          "fbcom4" : {"cy" : 53, "cdrop" : [33, 39, 46]},
          "fbcom5" : {"cy" : 53, "cdrop" : [33, 39, 46]},
          "forestfires" : {"cy" : 12, "header" : True, "dummies" : [2, 3]},
          "turbine2011" : {"cy" : 9, "cdrop" : [10], "header" : True},
          "turbine2012" : {"cy" : 9, "cdrop" : [10], "header" : True},
          "turbine2013" : {"cy" : 9, "cdrop" : [10], "header" : True},
          "turbine2014" : {"cy" : 9, "cdrop" : [10], "header" : True},
          "turbine2015" : {"cy" : 9, "cdrop" : [10], "header" : True},
          "heart_failure" : {"cy" : 12, "header" : True},
          "naval_propulsion" : {"cy" : 16, "cdrop" : [17]},
          "aquatic_toxicity" : {"cy" : 8},
          "indoor" : {"cy" : 520, "cdrop" : [521, 522, 523], "header" : True},
          "news" : {"cy" : 60, "cdrop" : [0, 1], "header" : True},
          "parkinson" : {"cy" : 5, "cdrop" : [4]},
          "redwine" : {"cy" : 11, "header" : True},
          "whitewine" : {"cy": 11, "header" : True},
          "student-mat": {"cy" : 32, "cdrop" : [30, 31], "header" : True, "dummies" : [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22]},
          "student-por" : {"cy" : 32, "cdrop" : [30, 31], "header" : True, "dummies" : [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22]},
          "superconduct" : {"cy" : 81, "header" : True},
          "year" : {"cy" : 0},
          "temperature" : {"cy" : 23, "cdrop" : [1, 24], "header" : True},
          "communities" : {"cy" : 127, "cdrop" : [0, 1, 2, 3, 4]},
          "compressive" : {"cy" : 8, "header" : True},
          "residential" : {"cy" : 107, "cdrop" : [108], "header" : True},
          "carbon" : {"cy" : 5, "cdrop" : [6, 7], "header" : True},
          "yacht" : {"cy" : 6},
          "wave_adelaide" : {"cy" : 48},
          "wave_tasmania" : {"cy" : 48},
          "wave_perth" : {"cy" : 48},
          "wave_sydney" : {"cy" : 48}}


dX = {}
dY = {}
for dfname in dFrame.keys():
    X, Y = process(li.get(dfname), dFrame[dfname]["cy"], dFrame[dfname].get("cdrop", []),
                         dFrame[dfname].get("header", False), dFrame[dfname].get("dummies", []))
    dX[dfname] = X.to_numpy()
    dY[dfname] = Y.to_numpy()

