import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)

class FacialExpressionModel(object):

    EMOTIONS_LIST = ["angry", "disgust", "afraid", "happy", "neutral", "sad", "surprised"]

    def __init__(self):
        model_json_file = 'static/jsfile/model.json'
        model_weights_file = 'static/jsfile/model_weights.h5'
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        emoji = FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
        return emoji