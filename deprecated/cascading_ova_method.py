import numpy as np 
import pandas as pd 
import seaborn as sns 
from sklearn.metrics import recall_score,f1_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import datetime

'''
A Cascading Version of the One-Vs-All algorithm

'''

df=pd.read_csv("tabular_train.csv")

y=df.id
df.drop(["id"],axis=1,inplace=True)

classes=y.unique().tolist()
print(y.value_counts())



df_test=pd.read_csv("tabular_valid.csv")

models={}

df_copy=df.copy()
b1=y==321
b2=y==-11
b3=y==211
b4=y==2212

models=[GradientBoostingClassifier(),RandomForestClassifier(),AdaBoostClassifier(),DecisionTreeClassifier(),XGBClassifier()]
for model in models:
    df_pos=df_copy[np.logical_or(np.logical_or(b1,b4),b2)].index
    df_neg=df_copy[b3].index


    df_copy.loc[df_pos,"id"]=1
    df_copy.loc[df_neg,"id"]=0
    print(df_copy.id.value_counts())
    y_copy=df_copy.id

    y_copy=df_copy.id
    df_copy.drop(["id"],axis=1,inplace=True)

    print(datetime.datetime.now())
    model_1=model

    model_1.fit(df_copy,y_copy)


    model_2=model

    df_copy_2=df[np.logical_or(np.logical_or(b1,b4),b2)]
    df_pos=df_copy_2[np.logical_or(b1,b2)].index
    df_neg=df_copy_2[b4].index

    df_copy_2.loc[df_pos,"id"]=1
    df_copy_2.loc[df_neg,"id"]=0
    print(df_copy_2.id.value_counts())
    y_copy=df_copy_2.id
    model_2.fit(df_copy_2.drop(["id"],axis=1),y_copy)


    model_3=model

    df_copy_3=df[np.logical_or(b1,b2)]
    df_pos=df_copy_3[b2].index
    df_neg=df_copy_3[b1].index
    df_copy_3.loc[df_pos,"id"]=1
    df_copy_3.loc[df_neg,"id"]=0
    print(df_copy_3.id.value_counts())
    y_copy=df_copy_3.id
    model_3.fit(df_copy_3.drop(["id"],axis=1),y_copy)
    y_pred=[]
    y_test=df_test.id
    df_test.drop(["id"],axis=1,inplace=True)
    prob_1=[]
    prob_2=[]
    prob_3=[]

    for i in range(len(df_test)):
        probas=[]
        sample=df_test.iloc[i]
        proba_1=model_1.predict_proba([sample])
        prob_1.append(proba_1[0][1])
        if proba_1[0][1]>=0.5:
            proba_2=model_2.predict_proba([sample])
            prob_2.append(proba_2[0][1])
            if proba_2[0][1]>=0.5:
                proba_3=model_3.predict_proba([sample])
                prob_3.append(proba_3[0][1])
                if(proba_3[0][1]>0.5):
                    y_pred.append(321)
                else: 
                    y_pred.append(-11)

            else:
                y_pred.append(2212)
        else:
            y_pred.append(211)




    print(f"For the Model {type(model).__name__} classification_report(y_test,np.array(y_pred))")
    print(datetime.datetime.now())