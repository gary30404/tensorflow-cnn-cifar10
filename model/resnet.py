import tensorflow as tf

nets = {
    'res50': [3, 4, 6, 3],
    'res101': [3, 4, 23, 3],
    'res152': [3, 8, 36, 3]
}

def bottleneck(inputs, num_filters, strides, phase_train):

    expansion = 4
    conv1 = tf.layers.conv2d(inputs=inputs, filters=num_filters, kernel_size=[1, 1], padding='SAME', activation=None)
    bn1 = batch_normalization_layer(conv1, num_filters, phase_train)
    out1 = tf.nn.relu(bn1)
    conv2 = tf.layers.conv2d(inputs=out1, filters=num_filters, kernel_size=[3, 3], padding='SAME', strides=strides, activation=None)
    bn2 = batch_normalization_layer(conv2, num_filters, phase_train)
    out2 = tf.nn.relu(bn2)
    conv3 = tf.layers.conv2d(inputs=out2, filters=num_filters*expansion, kernel_size=[1, 1], padding='SAME', activation=None)
    bn3 = batch_normalization_layer(conv3, num_filters*expansion, phase_train)

    # check if correct
    if strides != 1 or inputs.shape[3] != expansion*num_filters:
        shortcut = tf.layers.conv2d(inputs=inputs, filters=num_filters*expansion, kernel_size=[1, 1], padding='SAME', strides=strides, activation=None)
        bn_shortcut = batch_normalization_layer(shortcut, num_filters*expansion, phase_train)
        out = bn3 + bn_shortcut
    else:
        # expand dimensions of channels
        inputs_expand = tf.layers.conv2d(inputs=inputs, filters=bn3.shape[3], kernel_size=[1, 1], padding='SAME', activation=None)
        out = bn3 + inputs_expand
    out = tf.nn.relu(out)

    return out

def ResNet(x, num_classes, phase_train, net):
    #x_image = tf.reshape(x, [-1, image_size, image_size, image_channels])
    inputs = x
    n = nets[net]

    conv1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=[3, 3], strides=1, padding='SAME')
    bn1 = batch_normalization_layer(conv1, 64, phase_train)
    out = tf.nn.relu(bn1)

    num_filters = 64
    stride = 1
    for num_blocks in n:
        strides = [stride] + [1]*(num_blocks-1)
        for s in strides:
            out = bottleneck(out, num_filters, s, phase_train)
        stride = 2
        num_filters *= 2

    out = tf.layers.average_pooling2d(out, pool_size=[4, 4], strides=1, padding='VALID')

    # flatten
    flat = tf.reshape(out, [-1, int(out.shape[1]*out.shape[2]*out.shape[3])])

    #softmax
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
