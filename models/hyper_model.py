from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
from spec import DataSpec
import json
import logging
from models import ExampleModel


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


class HyperModel:
    def __init__(self, model, hp_values, configs):
        assert (model in self.list_available_models()), "we only have {} models".format(self.list_available_models())
        self.configs = configs
        if configs['items_local_path'] and configs['labels_local_path']:
            pass
        elif configs['remote_dataset_id']:
            pass
        else:
            items, labels = get_example_data()

        init_model = ExampleModel(hp_values['input_size'], hp_values['learning_rate'])
        self.model = init_model.build()

        self.labels = labels
        self.items = resize_images(items, hp_values['input_size'])

    def run(self):
        history = self.model.fit(self.items, self.labels, epochs=self.configs['epochs'], validation_split=0.1)
        metrics = {'val_accuracy': history.history['val_accuracy'][-1].item()}
        return metrics

    @staticmethod
    def list_available_models():
        return ['retinanet', 'yolo', 'mobilenet', 'example_model']
