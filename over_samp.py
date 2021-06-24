import numpy as np 
import pandas as pd
from sklearn.metrics import classification_report
from smote_variants import SMOTE,G_SMOTE,MulticlassOversampling

def split_label(df):

    return df.drop(["id"],axis=1),df.id



d_oversamp={"SMOTE":SMOTE(n_jobs=-1),"G_SMOTE":G_SMOTE(n_jobs=-1)}
df=pd.read_csv("tabular_train.csv")
X,y=split_label(df)

for n,over_samp in d_oversamp.items():
    
    oversampler=MulticlassOversampling(over_samp)
    X_train_ov,y_train_ov=oversampler.sample(X,y)

    df_ov=pd.DataFrame(X_train_ov,columns=X.columns)
    df_ov["id"]=y_train_ov
    df_ov.to_csv(f"Train_Overampling_{n}.csv",index=False)
    






