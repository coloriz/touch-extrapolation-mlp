import tensorflow as tf
from model import model_fn
import json


regressor = tf.estimator.Estimator(model_fn=model_fn, model_dir='./models')
names = [name for name in regressor.get_variable_names() if name.startswith('dense') or name.startswith('batch')]
data = {}
for name in names:
    value = regressor.get_variable_value(name)
    print(name, value.shape)
    data[name] = value.tolist()

with open('weights.json', 'w') as f:
    json.dump(data, f)