import tensorflow as tf

def network(x, y, phase_train):

    IMAGE_SIZE = 32
    IMAGE_CHANNEL = 3
    NUM_CLASSES = 10

    x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
    with tf.variable_scope('conv1') as scope:
        conv = tf.layers.conv2d(
                                inputs=x_image,
                                filters=32,
                                kernel_size=[5, 5],
                                padding='SAME',
                                activation=tf.nn.relu
                                )
        conv_bn = batch_norm(conv, 32, phase_train)
        pool = tf.layers.max_pooling2d(
                                       conv_bn, 
                                       pool_size=[2, 2], 
                                       strides=2, 
                                       padding='SAME'
                                       )

    with tf.variable_scope('conv2') as scope:
        conv = tf.layers.conv2d(
                                inputs=pool,
                                filters=64,
                                kernel_size=[5, 5],
                                padding='SAME',
                                activation=tf.nn.relu
                                )
        conv_bn = batch_norm(conv, 64, phase_train)
        pool = tf.layers.max_pooling2d(
                                       conv_bn, 
                                       pool_size=[2, 2], 
                                       strides=2, 
                                       padding='SAME'
                                       )

    with tf.variable_scope('conv3') as scope:
        conv = tf.layers.conv2d(
                                inputs=pool,
                                filters=128,
                                kernel_size=[3, 3],
                                padding='SAME',
                                activation=tf.nn.relu
                                )
        pool = tf.layers.max_pooling2d(
                                       conv, 
                                       pool_size=[2, 2], 
                                       strides=2, 
                                       padding='SAME'
                                       )

    with tf.variable_scope('conv4') as scope:
        conv = tf.layers.conv2d(
                                inputs=pool,
                                filters=256,
                                kernel_size=[3, 3],
                                padding='SAME',
                                activation=tf.nn.relu
                                )
        pool = tf.layers.max_pooling2d(
                                       conv, 
                                       pool_size=[2, 2], 
                                       strides=2, 
                                       padding='SAME'
                                       )

    with tf.variable_scope('fc6') as scope:
        flat = tf.reshape(pool, [-1, pool.shape[1] * pool.shape[2] * pool.shape[3]])
        fc = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)
        drop = tf.layers.dropout(fc, rate=0.5)

    with tf.variable_scope('fc7') as scope:
        fc = tf.layers.dense(inputs=drop, units=512, activation=tf.nn.relu)
        outputs = tf.layers.dense(inputs=fc, units=NUM_CLASSES, activation=tf.nn.softmax, name=scope.name)


    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y))
    predict = tf.argmax(outputs, axis=1)
    acc = tf.equal(predict, tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(acc, tf.float32))

    return loss, outputs, predict, accuracy


def batch_norm(x, n_out, phase_train):

    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
