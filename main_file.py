from utils.all_utils import data_preparation, save_model
from utils.model import ANN_model
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def main(data,epochs):

    model = ANN_model(epochs=epochs)
    
    X_train,y_train,X_valid,y_valid,X_test,y_test = data_preparation(data)

    model.fit(X_train,y_train,X_valid,y_valid)

    model_clf = model.predict(X_test,y_test)

    save_model(model_clf)


if __name__== '__main__':
    EPOCHS = 30
    mnist = tf.keras.datasets.mnist

    main(data=mnist ,epochs=EPOCHS)