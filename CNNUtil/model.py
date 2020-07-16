from tensorflow.keras import models, layers

class Model:
    @staticmethod
    def build_model(intensity=256):
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(intensity,)))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(2, activation='softmax'))
        return model



