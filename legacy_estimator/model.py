import tensorflow as tf


def model_fn(features, labels, mode):
    """Model function for Touch Extrapolation"""
    def _add_dense_layers(inputs, units, use_bias=True, batch_norm=True):
        """Adds dense, BN layers"""
        x = tf.layers.dense(
            inputs=inputs,
            units=units,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )
        if batch_norm:
            x = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN)
        x = tf.nn.relu(x)
        return x

    def _add_output_layers(inputs, units):
        """Adds output layer"""
        x = tf.layers.dense(
            inputs=inputs,
            units=units,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )
        x = tf.nn.tanh(x)
        return x

    x_speed, y_speed = features['x_speed'], features['y_speed']
    x_scalar, y_scalar = features['x_scalar'], features['y_scalar']

    input_layer = tf.concat([x_speed, y_speed, x_scalar, y_scalar], axis=1)
    x = _add_dense_layers(input_layer, 8)
    output_layer = _add_output_layers(x, 2)

    predictions = {
        'x_delta': output_layer[:, 0],
        'y_delta': output_layer[:, 1]
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Compute loss
    loss = tf.losses.mean_squared_error(labels=labels, predictions=output_layer)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=0.001,
            optimizer='Adam',
            learning_rate_decay_fn=lambda lr, gs: tf.train.exponential_decay(lr, gs, 500, 0.95),
            summaries=['learning_rate', 'loss', 'gradients']
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics
    eval_metric_ops = {
        'rmse': tf.metrics.root_mean_squared_error(labels=labels, predictions=output_layer)
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
