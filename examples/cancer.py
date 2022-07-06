from ann2pmml import ann2pmml
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X = X.astype(np.float32)
y = y.astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y, random_state=42)

y_train_ohe = to_categorical(y_train)
y_test_ohe = to_categorical(y_test)

mms = MinMaxScaler()
X_train_scaled = mms.fit_transform(X_train)
X_test_scaled = mms.transform(X_test)

model = Sequential()
model.add(Dense(units=X_train_scaled.shape[1], input_shape=(X_train_scaled.shape[1],), activation='tanh'))
model.add(Dense(units=20, activation='tanh'))
model.add(Dense(units=5, activation='tanh'))
model.add(Dense(units=y_test_ohe.shape[1], activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd')
model.fit(X_train_scaled, y_train_ohe, nb_epoch=100, batch_size=1, verbose=1, validation_data=None)

params = {
    'copyright': 'lampda',
    'description': 'Simple Keras model for Iris dataset.',
    'model_name': 'Iris Model'
}
file_name = 'cancer.pmml'
ann2pmml(model, transformer=mms, file=file_name, **params)
