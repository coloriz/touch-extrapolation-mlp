import matplotlib.pyplot as plt
import numpy as np
from time import sleep
from create_dataset import *

plt.ion()
fig1 = plt.figure(1, (8, 5))
ax1 = fig1.subplots()
fig2 = plt.figure(2)
ax2 = fig2.subplots()

ax1.set_xlim(0, 1920)
ax1.set_ylim(0, 1200)
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
line1, = ax1.plot([], [], 'o-', markersize=2)
line2, = ax2.plot([], [], 'o-', markersize=2)

strokes = get_strokes(Task.PAINT)

for s in strokes:
    normalized = s.normalized
    line1.set_xdata(s.x)
    line1.set_ydata(s.y)
    line2.set_xdata(normalized.x)
    line2.set_ydata(normalized.y)
    fig1.canvas.draw()
    fig1.canvas.flush_events()
    fig2.canvas.draw()
    fig2.canvas.flush_events()
    plt.waitforbuttonpress()
