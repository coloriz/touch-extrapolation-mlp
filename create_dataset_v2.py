from pathlib import Path
import sys
import pickle
import numpy as np
from utils import *


BASE_PATH = Path('./data')
WINDOW_SIZE = int(sys.argv[1])


def get_strokes(task: Task) -> list:
    filename_format = ''
    if task is Task.FITTS:
        filename_format = 'FittsTasks-participant{}.txt'
    elif task is Task.PAINT:
        filename_format = 'PaintTasks-participant{}.txt'
    elif task is Task.WRITE:
        filename_format = 'WriteTasks-participant{}.txt'

    strokes = []

    # participant 1~8
    for i in range(1, 8 + 1):
        file_path = BASE_PATH / filename_format.format(i)
        with file_path.open('r') as f:
            events = f.readlines()
        # 각 라인 토큰화
        tokens = map(lambda e: e.split(';'), events)
        # 이벤트로 변환
        events = map(lambda t: TouchEvent(float(t[2]), float(t[3]), int(t[1]), TouchEvent.Type(int(t[5]))), tokens)

        # 길이가 14 이상인 stroke만 출력에 추가
        stroke = []
        for e in events:
            stroke.append(e)
            if e.event_type is TouchEvent.Type.END:
                if len(stroke) > 14:
                    strokes.append(Stroke(stroke))
                stroke = []

    return strokes


def create_dataset(strokes):
    # 생성한 features와 output vector를 담을 리스트
    input_vectors, output_vectors = [], []

    # 모든 stroke에 대해
    for stroke in strokes:
        for i in range(len(stroke) - WINDOW_SIZE - 1):
            s = stroke[i:i + WINDOW_SIZE + 1]   # window + 1 크기 만큼 자름
            input_stroke: Stroke = s[:-1]
            output_touchevent: TouchEvent = s[-1]
            # 정규화를 진행하기 전 정규화 팩터가 너무 작은 stroke은 제외
            if input_stroke.x.ptp() < 0.001 or input_stroke.y.ptp() < 0.001:
                continue
            x_ptp, y_ptp = input_stroke.x.ptp(), input_stroke.y.ptp()
            norm_stroke = input_stroke.normalized
            x_speed, y_speed = norm_stroke.first_derivative
            x_acc, y_acc = norm_stroke.second_derivative
            timestamp_delta = norm_stroke.timestamp_delta
            x_scale, y_scale = 1 / x_ptp, 1 / y_ptp
            input_vector = np.concatenate((x_speed, y_speed, x_acc, y_acc, timestamp_delta, [x_scale, y_scale]))\
                             .astype(np.float32)
            last_point_x, last_point_y = input_stroke.x[-1], input_stroke.y[-1]
            # unscaled output_touchevent
            output_vector = np.array([output_touchevent.x - last_point_x,
                                      output_touchevent.y - last_point_y], np.float32)
            input_vectors.append(input_vector)
            output_vectors.append(output_vector)

    input_vectors = np.array(input_vectors)
    output_vectors = np.array(output_vectors)
    return input_vectors, output_vectors


def stroke_to_input_vector(stroke: Stroke):
    x_ptp, y_ptp = stroke.x.ptp(), stroke.y.ptp()
    norm_stroke = stroke.normalized
    x_speed, y_speed = norm_stroke.first_derivative
    x_acc, y_acc = norm_stroke.second_derivative
    timestamp_delta = norm_stroke.timestamp_delta
    x_scale, y_scale = 1 / x_ptp, 1 / y_ptp
    input_vector = np.concatenate((x_speed, y_speed, x_acc, y_acc, timestamp_delta, [x_scale, y_scale]))

    return input_vector.astype(np.float32)


def input_vector_to_stroke(input_vector: np.ndarray):
    raise NotImplementedError


if __name__ == '__main__':
    data = {}
    print(f'window size: {WINDOW_SIZE}')
    data['input_size'] = {
        'speed': 2 * (WINDOW_SIZE - 1),
        'acc': 2 * (WINDOW_SIZE - 2),
        'timestamp_delta': WINDOW_SIZE - 1,
        'scale': 2
    }
    total_train_data, total_test_data = 0, 0
    for task in [Task.PAINT, Task.WRITE, Task.FITTS]:
        print(f'processing task: {task}')
        s = get_strokes(task)
        input_vectors, output_vectors = create_dataset(s)
        permutation = np.random.permutation(len(input_vectors))
        input_vectors = input_vectors[permutation]
        output_vectors = output_vectors[permutation]

        n = int(len(input_vectors) * 0.9)
        total_train_data += n
        total_test_data += len(input_vectors) - n
        print(f'generated train data: {n} / test data: {len(input_vectors) - n}')

        data[str(task)] = (input_vectors[:n], output_vectors[:n]), \
                          (input_vectors[n:], output_vectors[n:])

    print(f'total train data: {total_train_data} / total test data: {total_test_data}')
    print(f'input vector size: {input_vectors.shape[1]} / output vector size: {output_vectors.shape[1]}')
    print(f'detail input vector size: {data["input_size"]}\n')

    with open(f'datasets_v2/touch_data_ws{WINDOW_SIZE}.pkl', 'wb') as f:
        pickle.dump(data, f)
