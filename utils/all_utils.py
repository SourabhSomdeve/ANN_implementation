import imp
import tensorflow as tf
import os
import logging

logger = logging.getLogger(__name__)

def data_preparation(data):

    
    (X_train_full, y_train_full),(X_test, y_test) = data.load_data()

    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    X_test = X_test / 255.

    logger.info("---Data Preparation done----")

    return X_train,y_train,X_valid,y_valid,X_test,y_test


def save_model(model):

    logger.info("saving the trained model")
    model_dir = "models"
    filename = "model.h5"
    os.makedirs(model_dir, exist_ok=True) 
    model_name = os.path.join(model_dir,filename)

    model.save(model_name)

