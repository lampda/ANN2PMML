ann2pmml
==========

ann2pmml is an automated pmml exporter for neural network models (for supported models see bellow) into PMML text format which address
the problems mentioned bellow.

Storing predictive models using binary format (e.g. Pickle) may be dangerous from several perspectives - naming few:

* **binary compatibility**:you update the libraries and may not be able to open the model serialized with older version
* **dangerous code**: when you would use model made by someone else
* **interpretability**: model cannot be easily opened and reviewed by human
* etc.

In addition the PMML is able to persist scaling of the raw input features which helps gradient descent to run smoothly
through optimization space.

Installation
------------

To install ann2pmml, simply:

.. code-block:: bash

    $ pip install ann2pmml

Example
-------

Example on Iris data - for more examples see the examples folder.

.. code-block:: python

    from ann2pmml import ann2pmml
    from sklearn.datasets import load_iris
    import numpy as np
    from sklearn.cross_validation import train_test_split
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
    model.add(Dense(units=X_train.shape[1], input_shape=(X_train_scaled.shape[1],), activation='tanh'))
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



Params explained
----------------
- **estimator**: Keras/TF model to be exported as PMML (for supported models - see bellow).
- **transformer**: if provided (and it's supported - see bellow) then scaling is applied to data fields.
- **file**: name of the file where the PMML will be exported.
- **feature_names**: when provided and have same shape as input layer, then features will have custom names, otherwise generic names (x\ :sub:`0`\,..., x\ :sub:`n-1`\) will be used.
- **target_values**: when provided and have same shape as output layer, then target values will have custom names, otherwise generic names (y\ :sub:`0`\,..., y\ :sub:`n-1`\) will be used.
- **target_name**: when provided then target variable will have custom name, otherwise generic name **class** will be used.
- **copyright**: who is the author of the model.
- **description**: optional parameter that sets *description* within PMML document.
- **model_name**: optional parameter that sets *model_name* within PMML document.

What is supported?
------------------
- Models
    * keras.models.Sequential
- Activation functions
    * tanh
    * sigmoid/logistic
    * linear
    * softmax normalization on the output layer (with activation identity on output units)
- Scalers
    * sklearn.preprocessing.StandardScaler
    * sklearn.preprocessing.MinMaxScaler

License
-------

This software is licensed under MIT licence.

- https://opensource.org/licenses/MIT