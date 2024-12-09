from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
from sys import argv
seed = 123
np.random.seed(seed)
import tensorflow as tf
tf.random.set_seed(seed)



log = argv[1]
df_train = pd.read_csv("dataset/"+log+"/"+log+"_train_norm.csv", sep=",", header=0)
df_test = pd.read_csv("dataset/"+log+"/"+log+"_test_norm.csv", sep=",", header=0)

y_train = df_train[df_train.columns[-1]]
y_test = df_test[df_test.columns[-1]]

df_train = df_train.iloc[:, :-1]
df_test = df_test.iloc[:, :-1]

input_len = len(df_train.columns)

df_train = df_train.values.astype(float)
df_test = df_test.values.astype(float)

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(df_train, y_train, test_size=0.2, random_state=42, stratify=y_train, shuffle=True)


print("input_len",input_len)
input_img= Input(shape=(input_len,))



#encoder
encoded = Dense(units=input_len, activation='tanh')(input_img)
encoded = Dense(units=200, activation='tanh')(encoded)
encoded = Dense(units=160, activation='tanh')(encoded)
encoded = Dense(units=120, activation='tanh')(encoded)

#latent-space
encoded = Dense(units=80, activation='tanh')(encoded)

#decoder
decoded = Dense(units=120, activation='tanh')(encoded)
decoded = Dense(units=160, activation='tanh')(decoded)
decoded = Dense(units=200, activation='tanh')(decoded)
decoded = Dense(units=input_len, activation='tanh')(decoded)


autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

autoencoder.summary()
encoder.summary()

autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
autoencoder.fit(X_train, X_train,
                epochs=int(argv[2]),
                batch_size=int(argv[3]),
                shuffle=True,
                validation_data=(X_val, X_val))

encoder.save("dataset/"+log+"/"+log+"_encoder.h5")
encoded_train = encoder.predict(df_train)
encoded_test = encoder.predict(df_test)

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler.fit(encoded_train)

train_norm = scaler.transform(encoded_train)
test_norm = scaler.transform(encoded_test)

enc_train = pd.DataFrame(train_norm)
enc_test = pd.DataFrame(test_norm)

enc_train.insert(len(enc_train.columns), "classification", y_train, True)
enc_test.insert(len(enc_test.columns), "classification", y_test, True)

enc_train.to_csv("dataset/"+log+"/"+log+"_train_enc.csv", index=False)
enc_test.to_csv("dataset/"+log+"/"+log+"_test_enc.csv", index=False)