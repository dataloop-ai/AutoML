from tensorflow import keras
from tensorflow.keras import layers


class ExampleModel:

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