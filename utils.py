from enum import Enum, auto
import numpy as np
import copy


class Task(Enum):
    FITTS = auto()
    PAINT = auto()
    WRITE = auto()


class TouchEvent:
    class Type(Enum):
        START = 0
        END = 1
        MOVE = 2

    def __init__(self, x: float, y: float, timestamp: int, event_type: Type):
        self.x = x
        self.y = y
        self.timestamp = timestamp
        self.event_type = event_type

    def __repr__(self):
        return f'TouchEvent(x={self.x}, y={self.y}, timestamp={self.timestamp}, type={self.event_type})'


class Stroke:
    def __init__(self, touch_events):
        self._touch_events = touch_events
        self.x = np.array([e.x for e in touch_events], dtype=np.float32)
        self.y = np.array([e.y for e in touch_events], dtype=np.float32)
        self.timestamp = np.array([e.timestamp for e in touch_events], dtype=np.int32)

    def __getitem__(self, item):
        # TODO : _touchevents는 정규화 안하는 문제 있음
        if isinstance(item, slice):
            replica = copy.copy(self)
            replica._touch_events = replica._touch_events.__getitem__(item)
            replica.x = replica.x.__getitem__(item)
            replica.y = replica.y.__getitem__(item)
            replica.timestamp = replica.timestamp.__getitem__(item)
            return replica
        else:
            return self._touch_events[item]

    def __len__(self):
        return len(self.x)

    def __repr__(self):
        return f'Stroke(length={len(self)})'

    @property
    def timestamp_delta(self):
        return self.timestamp[1:] - self.timestamp[0:-1]

    @property
    def normalized(self):
        norm = copy.copy(self)
        norm.x = 0.9 * (norm.x - norm.x.min()) / norm.x.ptp() - 0.45
        norm.y = 0.9 * (norm.y - norm.y.min()) / norm.y.ptp() - 0.45
        return norm

    @property
    def first_derivative(self):
        x_speed = self.x[1:] - self.x[0:-1]
        y_speed = self.y[1:] - self.y[0:-1]
        return x_speed, y_speed

    @property
    def second_derivative(self):
        x_speed, y_speed = self.first_derivative
        x_acc = x_speed[1:] - x_speed[0:-1]
        y_acc = y_speed[1:] - y_speed[0:-1]
        return x_acc, y_acc
