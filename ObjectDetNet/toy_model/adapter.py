from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
from .example_model import ExampleModel
import os


def get_example_data():
    (x, y), (val_x, val_y) = keras.datasets.mnist.load_data()
    x = x.astype('float32') / 255.
    val_x = val_x.astype('float32') / 255.
    x = x[:10000]
    y = y[:10000]
    return x, y


def resize_images(images, new_size):
    new_images = []
    for img in images:
        res = cv2.resize(img, dsize=(new_size, new_size),
                         interpolation=cv2.INTER_CUBIC)
        new_images.append(res)
    return np.array(new_images)


class AdapterModel:
    def __init__(self, devices, model_specs, hp_values, final):
        self.model_specs = model_specs
        self.hp_values = hp_values

    def data_loader(self):
        items, labels = get_example_data()
        self.labels = labels
        self.items = items

    def preprocess(self):
        self.items = resize_images(self.items, self.hp_values['input_size'])

    def build(self):
        init_model = ExampleModel(self.hp_values['input_size'], self.hp_values['learning_rate'])
        self.toy_model = init_model.build()
        self.hp_values = self.hp_values

    def train(self):
        history = self.toy_model.fit(self.items, self.labels, epochs=self.model_specs['training_configs']['epochs'], validation_split=0.1)
        self.val_accuracy = {'val_accuracy': history.history['val_accuracy'][-1].item()}

    def get_checkpoint(self):
        return self.get_best_checkpoint()

    def get_metrics(self):
        return self.val_accuracy
