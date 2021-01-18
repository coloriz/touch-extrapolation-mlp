import matplotlib.pyplot as plt
import tensorflow as tf
from utils import *
from model import model_fn
import pickle


with open('./touch_data.pkl', 'rb') as f:
    data = pickle.load(f)

(paint_train, paint_test), (x_paint_train, y_paint_train), (x_paint_test, y_paint_test) = data['Task.PAINT']
(write_train, write_test), (x_write_train, y_write_train), (x_write_test, y_write_test) = data['Task.WRITE']
(fitts_train, fitts_test), (x_fitts_train, y_fitts_train), (x_fitts_test, y_fitts_test) = data['Task.FITTS']

paint_train, paint_test = [Stroke(s) for s in paint_train], [Stroke(s) for s in paint_test]
write_train, write_test = [Stroke(s) for s in write_train], [Stroke(s) for s in write_test]


paint_input_fn = tf.estimator.inputs.numpy_input_fn(
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
write_input_fn = tf.estimator.inputs.numpy_input_fn(
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


def main(unused_argv):
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    norm_fig, norm_ax = plt.subplots()
    ax.set_xlim(0, 1920)
    ax.set_ylim(0, 1200)
    norm_ax.set_xlim(-1, 1)
    norm_ax.set_ylim(-1, 1)

    input_stroke, = ax.plot([], [], 'o-', markersize=2)
    predicted_points, = ax.plot([], [], 'ro-', markersize=2)
    norm_input_stroke, = norm_ax.plot([], [], 'o-', markersize=2)
    norm_predicted_points, = norm_ax.plot([], [], 'ro-', markersize=4)

    regressor = tf.estimator.Estimator(model_fn=model_fn, model_dir='./models')
    predictions = regressor.predict(input_fn=write_input_fn)
    for original_stroke, pred in zip(write_test, predictions):
        stroke, label = original_stroke[:-1], original_stroke[-1:]
        x_min, y_min = stroke.x.min(), stroke.y.min()
        x_ptp, y_ptp = stroke.x.ptp(), stroke.y.ptp()
        norm_stroke = stroke.normalized
        last_point_x, last_point_y = norm_stroke.x[-1], norm_stroke.y[-1]
        norm_label_x = 0.9 * (label.x[0] - x_min) / x_ptp - 0.45
        norm_label_y = 0.9 * (label.y[0] - y_min) / y_ptp - 0.45
        norm_predicted_point_x, norm_predicted_point_y = last_point_x + pred['x_delta'], last_point_y + pred['y_delta']
        scaled_prediction = [(norm_predicted_point_x + 0.45) * x_ptp / 0.9 + x_min,
                             (norm_predicted_point_y + 0.45) * y_ptp / 0.9 + y_min]
        error = np.sqrt((scaled_prediction[0] - label.x[0]) ** 2 + (scaled_prediction[1] - label.y[0]) ** 2)

        ax.set_title(f'error : {error:.2f} px')
        input_stroke.set_data(original_stroke.x, original_stroke.y)
        predicted_points.set_data(scaled_prediction)
        norm_stroke_with_label_x = np.concatenate((norm_stroke.x, [norm_label_x]))
        norm_stroke_with_label_y = np.concatenate((norm_stroke.y, [norm_label_y]))
        norm_input_stroke.set_data(norm_stroke_with_label_x, norm_stroke_with_label_y)
        norm_predicted_points.set_data(norm_predicted_point_x, norm_predicted_point_y)
        plt.waitforbuttonpress()
    print(len(x_paint_test))
    print(len(predictions))
    print(predictions)


if __name__ == '__main__':
    main([])
