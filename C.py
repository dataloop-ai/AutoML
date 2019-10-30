from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np



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
    def __init__(self, trial, configs, model, data):

        self.configs = configs
        self.model = model
        self.data = data
        self.new_images = []
        init_model = myHyperModel(trial['input_size'], trial['learning_rate'])
        self.model = init_model.build()

        for img in self.data['images']:
            res = cv2.resize(img, dsize=(trial['input_size'], trial['input_size']), interpolation=cv2.INTER_CUBIC)
            self.new_images.append(res)
        self.new_images_array = np.array(self.new_images)
        pass
    def run(self):

        history = self.model.fit(self.new_images_array, self.data['labels'], epochs=self.configs['epochs'], validation_split=0.1)
        metrics = {}
        metrics['val_accuracy'] = history.history['val_accuracy'][-1]
        return metrics
