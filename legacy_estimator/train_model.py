import numpy as np
import tensorflow as tf
from utils import *
import pickle
from model import model_fn

tf.logging.set_verbosity(tf.logging.DEBUG)

with open('./touch_data.pkl', 'rb') as f:
    data = pickle.load(f)

(paint_train, paint_test), (x_paint_train, y_paint_train), (x_paint_test, y_paint_test) = data['Task.PAINT']
(write_train, write_test), (x_write_train, y_write_train), (x_write_test, y_write_test) = data['Task.WRITE']
(fitts_train, fitts_test), (x_fitts_train, y_fitts_train), (x_fitts_test, y_fitts_test) = data['Task.FITTS']

x_train = np.concatenate((x_paint_train, x_write_train, x_fitts_train), axis=0)
y_train = np.concatenate((y_paint_train, y_write_train, y_fitts_train), axis=0)

permutation = np.random.permutation(len(x_train))
x_train = x_train[permutation]
y_train = y_train[permutation]


train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={
        'x_speed': x_train[:, 0:7],
        'y_speed': x_train[:, 7:14],
        'x_scalar': x_train[:, 14:15],
        'y_scalar': x_train[:, 15:16]
    },
    y=y_train,
    batch_size=512,
    num_epochs=20,
    shuffle=True
)
paint_test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={
        'x_speed': x_paint_test[:, 0:7],
        'y_speed': x_paint_test[:, 7:14],
        'x_scalar': x_paint_test[:, 14:15],
        'y_scalar': x_paint_test[:, 15:16]
    },
    y=y_paint_test,
    num_epochs=1,
    shuffle=False
)
write_test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={
        'x_speed': x_write_test[:, 0:7],
        'y_speed': x_write_test[:, 7:14],
        'x_scalar': x_write_test[:, 14:15],
        'y_scalar': x_write_test[:, 15:16]
    },
    y=y_write_test,
    num_epochs=1,
    shuffle=False
)
fitts_test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={
        'x_speed': x_fitts_test[:, 0:7],
        'y_speed': x_fitts_test[:, 7:14],
        'x_scalar': x_fitts_test[:, 14:15],
        'y_scalar': x_fitts_test[:, 15:16]
    },
    y=y_fitts_test,
    num_epochs=1,
    shuffle=False
)


def main(unused_argv):
    regressor = tf.estimator.Estimator(model_fn=model_fn, model_dir='./models')
    regressor.train(input_fn=train_input_fn)

    paint_eval = regressor.evaluate(input_fn=paint_test_input_fn)
    write_eval = regressor.evaluate(input_fn=write_test_input_fn)
    fitts_eval = regressor.evaluate(input_fn=fitts_test_input_fn)
    print('============= rmse of each tests =============')
    print(f'  PAINT TASK = {paint_eval["rmse"]}')
    print(f'  WRITE TASK = {write_eval["rmse"]}')
    print(f'  FITTS TASK = {fitts_eval["rmse"]}')


if __name__ == '__main__':
    tf.app.run(main=main)