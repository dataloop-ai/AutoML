from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
from spec import DataSpec
import json
import logging


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

class myHyperModel:

    def __init__(self, img_size, lr):
        self.img_size = (img_size, img_size)
        self.lr = lr

    def build(self):
        model = keras.Sequential()
        model.add(layers.Flatten(input_shape=self.img_size))
        model.add(layers.Dense(10, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        return model


class Experiment:
    def __init__(self, hp_values, model, configs):
        self.model = model
        self.configs = configs
        init_model = myHyperModel(hp_values['input_size'], hp_values['learning_rate'])
        self.model = init_model.build()

        if configs['items_local_path'] and configs['labels_local_path']:
            pass
        elif configs['remote_dataset_id']:
            pass
        else:
            items, labels = get_example_data()

        self.labels = labels
        self.items = resize_images(items, hp_values['input_size'])

    def run(self):
        history = self.model.fit(self.items, self.labels, epochs=self.configs['epochs'], validation_split=0.1)
        metrics = {'val_accuracy': history.history['val_accuracy'][-1].item()}
        return metrics
