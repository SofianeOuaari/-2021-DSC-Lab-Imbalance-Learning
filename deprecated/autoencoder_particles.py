import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Input,Model
from keras.layers import Dense
import keras
import datetime
import pickle


'''

Creating Features Using the an Autoencoder by saving only the Encoder part after training
'''


scaler=MinMaxScaler()

df_train=pd.read_csv("tabular_train.csv")
df_val=pd.read_csv("tabular_valid.csv")


y_train=df_train.id
y_val=df_val.id
X_train=df_train.drop(["id"],axis=1)
X_val=df_val.drop(["id"],axis=1)
shape=len(X_train.columns)
print(X_train.shape)
X_train=scaler.fit_transform(X_train)
X_val=scaler.transform(X_val)
filename = 'scaler.sav'
pickle.dump(scaler, open(filename, 'wb'))



input = Input(shape=(shape,))
encoded = Dense(512, activation='relu')(input)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(shape, activation='sigmoid')(decoded)

autoencoder = Model(input, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()

log_dir = "logs"

autoencoder.fit(X_train,X_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(X_val,X_val),
                )




model=Model(input,encoded)

model.save('encoder_1.keras')
