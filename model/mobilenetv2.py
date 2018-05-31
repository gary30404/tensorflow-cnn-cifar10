import tensorflow as tf

# (expansion, num_filters, num_blocks, stride)
nets = [(1,  16, 1, 1), (6,  24, 2, 1), (6,  32, 3, 2), (6,  64, 4, 2), (6,  96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

def bottleneck(inputs, num_filters_out, expansion, strides, phase_train):

    num_filters = expansion*inputs.shape[3]
    conv1 = tf.layers.conv2d(inputs=inputs, filters=num_filters, kernel_size=[1, 1], strides=1, padding='SAME', activation=None)
    bn1 = batch_normalization_layer(conv1, num_filters, phase_train)
    out1 = tf.nn.relu(bn1)
    conv2 = tf.layers.conv2d(inputs=out1, filters=num_filters, kernel_size=[3, 3], strides=strides, padding='SAME', activation=None)
    bn2 = batch_normalization_layer(conv2, num_filters, phase_train)
    out2 = tf.nn.relu(bn2)
    conv3 = tf.layers.conv2d(inputs=out1, filters=num_filters_out, kernel_size=[1, 1], strides=1, padding='SAME', activation=None)
    bn3 = batch_normalization_layer(conv3, num_filters, phase_train)

    if strides == 1 and inputs.shape[3] != num_filters_out:
        shortcut = tf.layers.conv2d(inputs=inputs, filters=num_filters_out, kernel_size=[1, 1], strides=1, padding='SAME', activation=None)
        bn_shortcut = batch_normalization_layer(shortcut, num_filters_out, phase_train)
        out = bn3 + bn_shortcut
    else:
       out = bn3

    return out

def MobileNetV2(x, image_size, image_channels, num_classes, phase_train, net):
    x_image = tf.reshape(x, [-1, image_size, image_size, image_channels])
    inputs = x_image

    conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[3, 3], strides=1, padding='SAME')
    bn1 = batch_normalization_layer(conv1, 32, phase_train)
    out = tf.nn.relu(bn1)
    for expansion, num_filters, num_blocks, stride in nets:
        strides = [stride] + [1]*(num_blocks-1)
        for stride in strides:
            out = bottleneck(out, num_filters, expansion, stride, phase_train)
    out = tf.layers.conv2d(out, filters=1280, kernel_size=[1, 1], strides=1, padding='SAME', activation=None)
    bn = batch_normalization_layer(out, 1280, phase_train)
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
