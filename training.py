import os
import time
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from MUIIA.DL_AEPIA.models import create_resnet, create_vgg, create_adhoc
from MUIIA.DL_AEPIA.config import FOLDER, MODEL, STEPS_PER_EPOCH, EPOCHS, BATCH_SIZE
import matplotlib.pyplot as plt


def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Error cuadrático medio')
  plt.plot(history.epoch, np.array(history.history['loss']),
           label='Error (loss) en entrenamiento')
  plt.plot(history.epoch, np.array(history.history['val_loss']),
           label='Valid Error (valid loss) en entrenamiento')
  plt.legend()
  plt.ylim([0, max(0.01, max(np.array(history.history['loss'])))])
  plt.show()


def preprocess_data(folder):
    all_images_path = os.listdir(os.path.join(folder, "images"))
    all_labels = pd.read_csv(os.path.join(folder, "percentage.csv"))
    print("Disponemos de {} imágenes".format(len(all_images_path)))
    print("Disponemos de {} labels en el csv".format(len(all_labels)))

    absolut_img_paths = [os.path.join(folder, "images", img_name) for img_name in all_labels["image_name"]]
    percentages = [value for value in all_labels["food_pixels"]]
    start = time.time()
    images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2BGR) for path in absolut_img_paths]
    images = [cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2))) for img in images]
    images = np.array(images)
    print("Ha tardado {}s en cargar las {} imágenes".format(round(time.time() - start, 2), len(absolut_img_paths)))
    print("\nTamaño de las imágenes: {}".format(np.shape(images[0])))
    print("\nPocentaje máximo de comida en el dataset: {} %".format(np.max(percentages)))
    print("Pocentaje mínimo de comida en el dataset: {} %".format(np.min(percentages)))
    print("Pocentaje medio de comida en el dataset: {} %".format(np.mean(percentages)))

    labels = np.array(percentages).astype(np.float32)

    return images, labels


if __name__ == "__main__":
    start = time.time()
    images, labels = preprocess_data(folder=FOLDER)

    # Split the data
    x_train = images[:int(0.7*len(images))]
    x_valid = images[int(0.7*len(images)):int(0.85*len(images))]
    x_test = images[int(0.85*len(images)):]
    y_train = labels[:int(0.7*len(labels))]
    y_valid = labels[int(0.7*len(labels)):int(0.85*len(labels))]
    y_test = labels[int(0.85*len(labels)):]

    train_datagen = ImageDataGenerator(horizontal_flip=True,
                                       vertical_flip=True,
                                       channel_shift_range=0.3,
                                       brightness_range=(0.6, 1))
    valid_datagen = ImageDataGenerator(horizontal_flip=True,
                                       vertical_flip=True,
                                       channel_shift_range=0.3,
                                       brightness_range=(0.6, 1))
    train_datagen.fit(x_train)
    valid_datagen.fit(x_valid)

    # Callbacks
    my_callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=12),
                   tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10,)]

    if MODEL == "resnet":
        model = create_resnet(np.shape(images[0]))
    elif MODEL == "vgg":
        model = create_vgg(np.shape(images[0]))
    elif MODEL == "adhoc":
        model = create_adhoc(np.shape(images[0]))
    else:
        model = create_resnet(np.shape(images[0]))

    model.compile(loss=tf.keras.metrics.mean_squared_error,
                  optimizer=keras.optimizers.Adam(learning_rate=1e-5),)

    history = model.fit(train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        validation_data=valid_datagen.flow(x_valid, y_valid),
                        epochs=EPOCHS, verbose=1, steps_per_epoch=STEPS_PER_EPOCH,
                        callbacks=my_callback)

    plot_history(history)
    results = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
    print(" Test loss: {}" .format(results))
    # Generate predictions
    # predictions = model.predict(x_test)
    print("It took {}s to train the model".format(time.time()-start))
