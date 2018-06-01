import tensorflow as tf


def bottleneck(inputs, growth_rate, phase_train):

    bn1 = batch_normalization_layer(inputs, inputs.shape[3], phase_train)
    bn1 = tf.nn.relu(bn1)
    conv1 = tf.layers.conv2d(inputs=inputs, filters=4*growth_rate, kernel_size=[1, 1], padding='SAME', activation=None)
    bn2 = batch_normalization_layer(conv1, 4*growth_rate, phase_train)
    bn2 = tf.nn.relu(bn2)
    conv2 = tf.layers.conv2d(inputs=bn2, filters=growth_rate, kernel_size=[3, 3], padding='SAME', activation=None)

    return tf.concat([conv2, inputs], 3)

def transition(inputs, num_filters, phase_train):

    bn = batch_normalization_layer(inputs, inputs.shape[3], phase_train)
    bn = tf.nn.relu(bn)
    conv = tf.layers.conv2d(inputs=bn, filters=num_filters, kernel_size=[1, 1], padding='SAME', activation=None)
    out = tf.layers.average_pooling2d(conv, pool_size=[2, 2], strides=2, padding='VALID')
    return out

def DenseNet(x, image_size, image_channels, num_classes, phase_train, net):
    x_image = tf.reshape(x, [-1, image_size, image_size, image_channels])
    inputs = x_image
    growth_rate = 12
    reduction = 0.5
    nblocks = [6,12,24,16]

    num_filters = 2*growth_rate
    out = tf.layers.conv2d(inputs=inputs, filters=num_filters, kernel_size=[3, 3], padding='SAME')
    for i in range(6):
        out = bottleneck(out, growth_rate, phase_train)
    num_filters += 6*growth_rate
    num_filters_out = int(math.floor(num_filters*reduction))
    out = transition(out, num_filters_out, phase_train)
    num_filters = num_filters_out
    for i in range(12):
        out = bottleneck(out, growth_rate, phase_train)
    num_filters += 12*growth_rate
    num_filters_out = int(math.floor(num_filters*reduction))
    out = transition(out, num_filters_out, phase_train)
    num_filters = num_filters_out
    for i in range(24):
        out = bottleneck(out, growth_rate, phase_train)
    num_filters += 24*growth_rate
    num_filters_out = int(math.floor(num_filters*reduction))
    out = transition(out, num_filters_out, phase_train)
    num_filters = num_filters_out
    for i in range(16):
        out = bottleneck(out, growth_rate, phase_train)
    num_filters += 16*growth_rate
    bn = batch_normalization_layer(out, num_filters, phase_train)
    out = tf.nn.relu(bn)

    out = tf.layers.average_pooling2d(out, pool_size=[4, 4], strides=1, padding='VALID')
    flat = tf.reshape(out, [-1, int(out.shape[1]*out.shape[2]*out.shape[3])])
    outputs = tf.layers.dense(inputs=flat, units=num_classes, activation=tf.nn.softmax)
    return outputs

def batch_normalization_layer(x, n_out, phase_train):
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        if len(x.shape) == 4:
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        else:
            batch_mean, batch_var = tf.nn.moments(x, [0,1], name='moments')
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
