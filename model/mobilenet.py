import tensorflow as tf

nets = [(64,1), (128,2), (128,1), (256,2), (256,1), (512,2), (512,1), (512,1), (512,1), (512,1), (512,1), (1024,2), (1024,1)]

def bottleneck(inputs, num_filters, strides, phase_train):

    conv1 = tf.layers.conv2d(inputs=inputs, filters=inputs.shape[3], kernel_size=[3, 3], strides=strides, padding='SAME', activation=None)
    bn1 = batch_normalization_layer(conv1, inputs.shape[3], phase_train)
    out1 = tf.nn.relu(bn1)
    conv2 = tf.layers.conv2d(inputs=out1, filters=num_filters, kernel_size=[1, 1], strides=1, padding='SAME', activation=None)
    bn2 = batch_normalization_layer(conv2, num_filters, phase_train)
    out2 = tf.nn.relu(bn2)

    return out2

def MobileNet(x, image_size, image_channels, num_classes, phase_train, net):
    x_image = tf.reshape(x, [-1, image_size, image_size, image_channels])
    inputs = x_image

    conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[3, 3], strides=1, padding='SAME')
    bn1 = batch_normalization_layer(conv1, 32, phase_train)
    out = tf.nn.relu(bn1)
    for layers in nets:
        num_filters = layers[0]
        strides = layers[1]
        out = bottleneck(out, num_filters, strides, phase_train)
    out = tf.layers.average_pooling2d(out, pool_size=[2, 2], strides=1, padding='VALID')
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
