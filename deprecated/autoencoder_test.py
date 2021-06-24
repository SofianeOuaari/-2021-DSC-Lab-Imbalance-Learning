import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import keras

'''
Comparing Two XGBoost models with and without the learned features from the Autoencoder

'''
df=pd.read_csv("pid-5M.csv").iloc[:3000000]
scaler=MinMaxScaler()

y=df.id
df.drop(["id"],axis=1,inplace=True)
columns=df.columns
df_scaled=pd.DataFrame(scaler.fit_transform(df),columns=columns)
model=keras.models.load_model("encoder.keras")
df_added_features=pd.DataFrame(model.predict(df_scaled))
df_scaled_original=df_scaled.copy()
df_scaled[df_added_features.columns]=df_added_features


X_train,X_test,y_train,y_test=train_test_split(df_scaled_original,y,stratify=y,test_size=0.25,random_state=110)
X_train_fea,X_test_fea,y_train_fea,y_test_fea=train_test_split(df_added_features,y,stratify=y,test_size=0.25,random_state=110)

clf1=XGBClassifier()
clf2=XGBClassifier()

clf1.fit(X_train,y_train)
clf2.fit(X_train_fea,y_train_fea)


y_pred_1=clf1.predict(X_test)
y_pred_2=clf2.predict(X_test_fea)

print(classification_report(y_test,y_pred_1))
print(classification_report(y_test_fea,y_pred_2))

