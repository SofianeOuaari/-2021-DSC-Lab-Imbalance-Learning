import numpy as np 
import pandas as pd


def adding_features_v1(train,valid,test):
    '''Applying Some Feature Engineering to Withdraw Implicit Patterns'''

    df_train=pd.read_csv(train) 
    df_valid=pd.read_csv(valid) 
    df_test=pd.read_csv(test)

    df_train["e_diff"]=df_train["eout"]-df_train["ein"]
    df_train["e_ratio"]=df_train["eout"]/df_train["ein"]
    df_train["e_sum"]=df_train["eout"]+df_train["ein"]
    df_train["nphe_0"]=(df_train["nphe"]==0).astype("int")
    df_train["ein_0"]=(df_train["ein"]==0).astype("int")
    df_train["eout_0"]=(df_train["eout"]==0).astype("int")

    df_valid["e_diff"]=df_valid["eout"]-df_valid["ein"]
    df_valid["e_ratio"]=df_valid["eout"]/df_valid["ein"]
    df_valid["e_sum"]=df_valid["eout"]+df_valid["ein"]
    df_valid["nphe_0"]=(df_valid["nphe"]==0).astype("int")
    df_valid["ein_0"]=(df_valid["ein"]==0).astype("int")
    df_valid["eout_0"]=(df_valid["eout"]==0).astype("int")

    df_test["e_diff"]=df_test["eout"]-df_test["ein"]
    df_test["e_ratio"]=df_test["eout"]/df_test["ein"]
    df_test["e_sum"]=df_test["eout"]+df_test["ein"]
    df_test["nphe_0"]=(df_test["nphe"]==0).astype("int")
    df_test["ein_0"]=(df_test["ein"]==0).astype("int")
    df_test["eout_0"]=(df_test["eout"]==0).astype("int")



    df_train.to_csv("train_added_features_4.csv",index=False)
    df_valid.to_csv("valid_added_features_4.csv",index=False)
    df_test.to_csv("test_added_features_4.csv",index=False)

def adding_features_v1_any(f):
    df=pd.read_csv(f) 
    

    df["e_diff"]=df["eout"]-df["ein"]
    df["e_ratio"]=df["eout"]/df["ein"]
    df["e_sum"]=df["eout"]+df["ein"]
    df["nphe_0"]=(df["nphe"]==0).astype("int")
    df["ein_0"]=(df["ein"]==0).astype("int")
    df["eout_0"]=(df["eout"]==0).astype("int")


    f=f.replace(".csv","")
    df.to_csv(f"{f}_added_features_1.csv",index=False)
    


def log_transformations(train,valid,test):
    t=pd.read_csv(train).id
    v=pd.read_csv(valid).id
    te=pd.read_csv(test).id
    df_train=pd.read_csv(train).drop(["id"],axis=1) 
    df_valid=pd.read_csv(valid).drop(["id"],axis=1) 
    df_test=pd.read_csv(test).drop(["id"],axis=1) 
    
    original_columns=df_train.columns
    for col in df_train.columns:
        df_train[f"{col}_log"]=np.log(df_train[col]+1)
        df_valid[f"{col}_log"]=np.log(df_valid[col]+1)
        df_test[f"{col}_log"]=np.log(df_test[col]+1)
    df_train.drop(original_columns,axis=1,inplace=True)
    df_valid.drop(original_columns,axis=1,inplace=True)
    df_test.drop(original_columns,axis=1,inplace=True)
    df_train["id"]=t
    df_valid["id"]=v
    df_test["id"]=te

    df_train.to_csv("tabular_train_log_4.csv",index=False)
    df_valid.to_csv("tabular_valid_log_4.csv",index=False)
    df_test.to_csv("tabular_test_log_4.csv",index=False)

def log_transformations_any(f):
    df=pd.read_csv(f) 
    y=df.id
    df.drop(["id"],axis=1,inplace=True)
    
    original_columns=df.columns
    for col in df.columns:
        df[f"{col}_log"]=np.log(df[col]+1)
       
    df.drop(original_columns,axis=1,inplace=True)
    df["id"]=y
    f=f.replace(".csv","")
    df.to_csv("active_learning_log.csv",index=False)
   


log_transformations("tabular_train.csv","tabular_valid.csv","tabular_test.csv")
adding_features_v1("tabular_train.csv","tabular_valid.csv","tabular_test.csv")
adding_features_v1_any("active_learning_4.csv")
log_transformations_any("active_learning_4.csv")
