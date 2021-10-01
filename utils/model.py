from contextlib import nullcontext
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

class ANN_model():

    def __init__(self,epochs):
        self.epochs = epochs
        self.model_clf = None
        
        
    def fit(self,X_train,y_train,X_valid,y_valid):

        LAYERS = [
                tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
                tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
                tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
                tf.keras.layers.Dense(10, activation="softmax", name="outputLayer")
        ]

        self.model_clf = tf.keras.models.Sequential(LAYERS)

        LOSS_FUNCTION = "sparse_categorical_crossentropy"
        OPTIMIZER = "SGD"
        METRICS = ["accuracy"]


        self.model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)

    
        VALIDATION = (X_valid, y_valid)

        history = self.model_clf.fit(X_train, y_train, epochs=self.epochs, validation_data=VALIDATION)



    def predict(self,X_test,y_test):


        self.model_clf.evaluate(X_test, y_test)

        X_new = X_test[:3]

        y_prob = self.model_clf.predict(X_new)

        y_prob.round(3)

        Y_pred= np.argmax(y_prob, axis=-1)
        Y_pred

        for img_array, pred, actual in zip(X_new, Y_pred, y_test[:3]):
        plt.imshow(img_array, cmap="binary")
        plt.title(f"predicted: {pred}, Actual: {actual}")
        plt.axis("off")
        plt.show()
        print("---"*20)

    