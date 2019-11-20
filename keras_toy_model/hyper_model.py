from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
from .example_model import ExampleModel
import os


def get_example_data():
    (x, y), (val_x, val_y) = keras.datasets.mnist.data_loader()
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


class HyperModel:
    def __init__(self, model, hp_values):
        self.model = model
        self.hp_values = hp_values

    def data_loader(self):
        items, labels = get_example_data()
        self.labels = labels
        self.items = items

    def add_preprocess(self, hp_values):
        self.items = resize_images(self.items, hp_values['input_size'])

    def build(self, hp_values):
        init_model = ExampleModel(hp_values['input_size'], hp_values['learning_rate'])
        self.model = init_model.build()
        self.hp_values = hp_values

    def train(self):
        history = self.model.fit(self.items, self.labels, epochs=self.model['training_configs']['epochs'], validation_split=0.1)
        train_metrics = {'val_accuracy': history.history['val_accuracy'][-1].item()}
        return train_metrics

    def infer(self):
        pass

    def eval(self):
        pass

    @staticmethod
    def list_available_models():
        return ['retinanet', 'yolo', 'mobilenet', 'example_model']
