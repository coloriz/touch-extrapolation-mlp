import pickle

import numpy as np
import tensorflow as tf

from utils import *


def train_model(data):
    input_size = data['input_size']
    (x_paint_train, y_paint_train), (x_paint_test, y_paint_test) = data['Task.PAINT']
    (x_write_train, y_write_train), (x_write_test, y_write_test) = data['Task.WRITE']
    (x_fitts_train, y_fitts_train), (x_fitts_test, y_fitts_test) = data['Task.FITTS']

    x_train = np.concatenate((x_paint_train, x_write_train, x_fitts_train), axis=0)
    y_train = np.concatenate((y_paint_train, y_write_train, y_fitts_train), axis=0)

    # shuffle data
    permutation = np.random.permutation(len(x_train))
    x_train = x_train[permutation]
    y_train = y_train[permutation]

    x_test = np.concatenate((x_paint_test, x_write_test, x_fitts_test), axis=0)
    y_test = np.concatenate((y_paint_test, y_write_test, y_fitts_test), axis=0)

    speed = tf.keras.layers.Input(shape=(input_size['speed'],))
    acc = tf.keras.layers.Input(shape=(input_size['acc'],))
    timestamp_delta = tf.keras.layers.Input(shape=(input_size['timestamp_delta'],))
    scale = tf.keras.layers.Input(shape=(input_size['scale'],))

    y = tf.keras.layers.concatenate([speed, acc, timestamp_delta, scale])
    y = tf.keras.layers.Dense(64)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Dense(32)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Dense(2)(y)
    y = tf.keras.layers.Activation('tanh')(y)
    y = (y + 0.45) / scale / 0.9

    model = tf.keras.Model(inputs=[speed, acc, timestamp_delta, scale], outputs=y, name='touch_exptrapolation_model')

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(64, input_shape=(x_train.shape[1],)),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.ReLU(),
    #     tf.keras.layers.Dense(32),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.ReLU(),
    #     tf.keras.layers.Dense(2, activation='tanh')
    # ])
    model.summary()
    tf.keras.utils.plot_model(model, './touch_extrapolation_model.png', show_shapes=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.fit(x_train, y_train,
              batch_size=512,
              epochs=3,
              validation_data=(x_test, y_test))
    result = {
        Task.PAINT: model.evaluate(x_paint_test, y_paint_test, batch_size=1024),
        Task.WRITE: model.evaluate(x_write_test, y_write_test, batch_size=1024),
        Task.FITTS: model.evaluate(x_fitts_test, y_fitts_test, batch_size=1024)
    }

    return result


WINDOW_SIZES = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
results = []
for i in WINDOW_SIZES:
    with open(f'./datasets_v2/touch_data_ws{i}.pkl', 'rb') as f:
        data = pickle.load(f)

    result = train_model(data)
    results.append((i, result))

for r in results:
    window_size, result = r
    print(f'================== WINDOW SIZE: {window_size} ==================')
    print(f'= PAINT: loss={result[Task.PAINT][0]}\trmse={result[Task.PAINT][1]}')
    print(f'= WRITE: loss={result[Task.WRITE][0]}\trmse={result[Task.WRITE][1]}')
    print(f'= FITTS: loss={result[Task.FITTS][0]}\trmse={result[Task.FITTS][1]}')
    print('')

# window size: 3 / result: [160.79535556572347, 12.680511]
# window size: 4 / result: [20.957727118276217, 4.577961]
# window size: 5 / result: [17.4609953830502, 4.1786356]
# window size: 6 / result: [0.37063228049861024, 0.6087957]
# window size: 7 / result: [1.5981865595120197, 1.264194]
# window size: 8 / result: [0.5454792977611914, 0.73856556]
# window size: 9 / result: [0.0722704185679726, 0.26883155]
# window size: 10 / result: [0.11683623238033354, 0.34181315]
# window size: 11 / result: [0.03915322816813882, 0.19787174]
# window size: 12 / result: [0.5410282432698946, 0.7355462]
# window size: 13 / result: [0.027136573432041098, 0.16473183]
