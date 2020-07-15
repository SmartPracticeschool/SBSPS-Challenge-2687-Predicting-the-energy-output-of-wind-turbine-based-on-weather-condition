import numpy as np
import keras.models
from keras.models import model_from_json
import tensorflow as tf


def init():
    json_file = open('regressor.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    loaded_model.compile(optimizer='adam', loss='mean_squared_error')

    graph = tf.compat.v1.get_default_graph()

    return loaded_model, graph

