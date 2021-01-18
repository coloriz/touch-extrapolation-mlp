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
                if len(stroke) > 16:
                    strokes.append(Stroke(stroke))
                stroke = []

    return strokes


def create_dataset(strokes):
    # 생성한 features와 output vector를 담을 리스트
    input_vectors, output_vectors = [], []

    # 모든 stroke에 대해
    for stroke in strokes:
        for i in range(len(stroke) - WINDOW_SIZE - 4):
            s = stroke[i:i + WINDOW_SIZE + 4]   # window 크기 만큼 자름
            input_stroke: Stroke = s[:-4]
            last_point_x, last_point_y = input_stroke.x[-1], input_stroke.y[-1]
            ov1: TouchEvent = s[-4]
            ov2: TouchEvent = s[-3]
            ov3: TouchEvent = s[-2]
            ov4: TouchEvent = s[-1]
            # 정규화 팩터가 너무 작은 stroke은 제외
            if input_stroke.x.ptp() < 0.001 or input_stroke.y.ptp() < 0.001:
                continue

            output_vector = np.array([[ov1.x - last_point_x, ov1.y - last_point_y],
                                      [ov2.x - ov1.x, ov2.y - ov1.y],
                                      [ov3.x - ov2.x, ov3.y - ov2.y],
                                      [ov4.x - ov3.x, ov4.y - ov3.y]], np.float32)
            input_vectors.append(input_stroke)
            output_vectors.append(output_vector)

    input_vectors = np.array(input_vectors)
    output_vectors = np.array(output_vectors)
    return input_vectors, output_vectors


if __name__ == '__main__':
    data = {}
    print(f'window size: {WINDOW_SIZE}')
    total_size = 0
    for task in [Task.PAINT, Task.WRITE, Task.FITTS]:
        print(f'processing task: {task}')
        s = get_strokes(task)
        input_vectors, output_vectors = create_dataset(s)
        permutation = np.random.permutation(len(input_vectors))
        input_vectors = input_vectors[permutation]
        output_vectors = output_vectors[permutation]

        n = len(input_vectors)
        total_size += n
        print(f'generated data: {n}')

        data[str(task)] = (input_vectors, output_vectors)

    print(f'total data: {total_size}')
    print(f'input vector size: {input_vectors.shape[1]} / output vector size: {output_vectors.shape[1]}\n')

    with open(f'test_datasets/touch_data_ws{WINDOW_SIZE}.pkl', 'wb') as f:
        pickle.dump(data, f)
