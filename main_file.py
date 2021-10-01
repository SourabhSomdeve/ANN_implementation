from utils.all_utils import data_preparation
from.utils.model import ANN_model
import tensorflow as tf




def main(data,epochs):

    model = ANN_model(epochs=epochs)
    
    X_train,y_train,X_valid,y_valid,X_test,y_test = data_preparation(data)

    model_ = model.fit(X_train,y_train,X_valid,y_valid)



if __name__== '__main__':
    EPOCHS = 30
    mnist = tf.keras.datasets.mnist

    main(data=mnist ,epochs=EPOCHS)