from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
from spec import DataSpec
import json
import logging

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
    def __init__(self, hp_values, model, configs, items, labels):

        new_images = []
        self.model = model
        self.configs = configs
        self.labels = labels
        init_model = myHyperModel(hp_values['input_size'], hp_values['learning_rate'])
        self.model = init_model.build()

        for img in items:
            res = cv2.resize(img, dsize=(hp_values['input_size'], hp_values['input_size']), interpolation=cv2.INTER_CUBIC)
            new_images.append(res)
        new_images_array = np.array(new_images)
        self.new_items = new_images_array

    def run(self):

        history = self.model.fit(self.new_items, self.labels, epochs=self.configs['epochs'], validation_split=0.1)
        logging.info('history')
        logging.info(history.history)
        metrics = {'val_accuracy': history.history['val_accuracy'][-1].item()}
        return metrics
