from ann2pmml import ann2pmml
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

iris = load_iris()
X = iris.data
y = iris.target

X = X.astype(np.float32)
y = y.astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

std = StandardScaler()
X_train_scaled = std.fit_transform(X_train)
X_test_scaled = std.transform(X_test)
y_train_ohe = to_categorical(y_train)
y_test_ohe = to_categorical(y_test)

model = Sequential()
model.add(Dense(units=X_train.shape[1], input_shape=(X.shape[1],), activation='tanh'))
model.add(Dense(units=5, activation='tanh'))
model.add(Dense(units=y_test_ohe.shape[1], activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='sgd')
model.fit(X_train_scaled, y_train_ohe, nb_epoch=10, batch_size=1, verbose=1,
          validation_data=(X_test_scaled, y_test_ohe))

params = {
    'feature_names': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    'target_values': ['setosa', 'virginica', 'versicolor'],
    'target_name': 'specie',
    'copyright': 'lampda',
    'description': 'Simple Keras model for Iris dataset.',
    'model_name': 'Iris Model'
}

ann2pmml(estimator=model, transformer=std, file='keras_iris.pmml', **params)
