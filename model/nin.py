import tensorflow as tf

def mlp(inputs, num_filters, phase_train):

    conv = tf.layers.conv2d(inputs=inputs, filters=num_filters, kernel_size=[3, 3], padding='SAME', strides=1)
    bn = batch_normalization_layer(conv, num_filters, phase_train)
    out = tf.nn.relu(bn)

    return out

def NiN(x, image_size, image_channels, num_classes, phase_train, net):
    x_image = tf.reshape(x, [-1, image_size, image_size, image_channels])
    inputs = x_image

    conv1 = tf.layers.conv2d(inputs=inputs, filters=192, kernel_size=[3, 3], strides=1, padding='SAME')
    bn1 = batch_normalization_layer(conv1, 192, phase_train)
    out1 = tf.nn.relu(bn1)
    mlp1_1 = mlp(out1, 160, phase_train)
    mlp1_2 = mlp(mlp1_1, 96, phase_train)
    
    out = tf.layers.max_pooling2d(mlp1_2, pool_size=[3, 3], strides=2, padding='SAME')

    conv2 = tf.layers.conv2d(inputs=out, filters=192, kernel_size=[3, 3], strides=1, padding='SAME')
    bn2 = batch_normalization_layer(conv2, 192, phase_train)
    out2 = tf.nn.relu(bn2)
    mlp2_1 = mlp(out2, 192, phase_train)
    mlp2_2 = mlp(mlp2_1, 192, phase_train)
    
    out = tf.layers.max_pooling2d(mlp2_2, pool_size=[3, 3], strides=2, padding='SAME')

    conv3 = tf.layers.conv2d(inputs=out, filters=192, kernel_size=[3, 3], strides=1, padding='SAME')
    bn3 = batch_normalization_layer(conv3, 192, phase_train)
    out3 = tf.nn.relu(bn3)

    mlp3_1 = mlp(out3, 192, phase_train)
    mlp3_2 = mlp(mlp3_1, 10, phase_train)

    out = tf.layers.average_pooling2d(mlp3_2, pool_size=[2, 2], strides=2, padding='VALID')
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
