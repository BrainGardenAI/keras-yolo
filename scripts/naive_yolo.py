"""
Simple yolo implementation with no configs
"""
from keras.layers import Input, Dense, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.layers.local import LocallyConnected2D


def buildYoloModel():
    inputs = Input(shape=(448, 448, 3))
    x = Conv2D(kernel_size=7, filters=64, strides=2, padding="same")(inputs)
    x=MaxPooling2D(strides=2, pool_size=2, padding="same")(x)
    # MP
    x = Conv2D(kernel_size=(3, 3), filters=192, padding="same")(x)
    x=MaxPooling2D(strides=2, pool_size=(2, 2), padding="same")(x)
    # MP
    x = Conv2D(kernel_size=1, filters=128, padding="same")(x)
    x = Conv2D(kernel_size=3, filters=256, padding="same")(x)
    x = Conv2D(kernel_size=1, filters=256, padding="same")(x)
    x = Conv2D(kernel_size=3, filters=256, padding="same")(x)
    x = MaxPooling2D(strides=2, pool_size=2, padding="same")(x)
    #MP
    for i in range(4):
        x = Conv2D(kernel_size=1, filters=256, padding="same")(x)
        x = Conv2D(kernel_size=3, filters=512, padding="same")(x)
    x = Conv2D(kernel_size=1, filters=512, padding="same")(x)
    x = Conv2D(kernel_size=3, filters=1024, padding="same")(x)
    x=MaxPooling2D(pool_size=2, strides=2, padding="same")(x)
    #MP
    for i in range(2):
        x = Conv2D(kernel_size=1, filters=512, padding="same")(x)
        x = Conv2D(kernel_size=3, filters=1024, padding="same")(x)
    x = Conv2D(kernel_size=3, filters=1024, padding="same")(x)
    x = Conv2D(kernel_size=3, filters=1024, strides=2, padding="same")(x)

    x = Conv2D(kernel_size=1, filters=1024, padding="same")(x)
    x = Conv2D(kernel_size=3, filters=1024, padding="same")(x)
    x = LocallyConnected2D(filters=256, kernel_size=3)(x)
    predictions = Dense(1715)(x)
    print(predictions.shape)
    return Model(inputs=inputs, outputs=predictions)
    

model = buildYoloModel()
