import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model


def relu_clipped(x):
    return K.relu(x, max_value=100)


def create_resnet(input_shape):

    resnet50 = keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

    for layer in resnet50.layers:
        layer.trainable = False  # por defecto, el valor de trainable es True

    x = resnet50.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    output = keras.layers.Dense(1, activation=relu_clipped)(x)

    resnet_model = Model(inputs=[resnet50.input], outputs=[output])
    resnet_model.summary()

    return resnet_model


def create_vgg(input_shape):
    vgg16 = keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    for layer in vgg16.layers:
        layer.trainable = False

    x = vgg16.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    output = keras.layers.Dense(1, activation=relu_clipped)(x)

    model = Model(inputs=[vgg16.input], outputs=[output])
    model.summary()

    return model


def create_adhoc(input_shape):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=input_shape,
                                     kernel_initializer="he_normal"))
    model.add(tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_initializer="he_normal"))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_initializer="he_normal"))
    model.add(tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_initializer="he_normal"))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Conv2D(256, 3, activation='relu', kernel_initializer="he_normal"))
    model.add(tf.keras.layers.Conv2D(256, 3, activation='relu', kernel_initializer="he_normal"))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer="he_normal"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer="he_normal"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1, activation=relu_clipped, kernel_initializer="he_normal"))
    model.summary()

    return model
