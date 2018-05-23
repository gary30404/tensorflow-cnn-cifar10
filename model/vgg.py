import tensorflow as tf

nets = {
    'VGG11': [64, 'p', 128, 'p', 256, 256, 'p', 512, 512, 'p', 512, 512, 'p'],
    'VGG13': [64, 64, 'p', 128, 128, 'p', 256, 256, 'p', 512, 512, 'p', 512, 512, 'p'],
    'VGG16': [64, 64, 'p', 128, 128, 'p', 256, 256, 256, 'p', 512, 512, 512, 'p', 512, 512, 512, 'p'],
    'VGG19': [64, 64, 'p', 128, 128, 'p', 256, 256, 256, 256, 'p', 512, 512, 512, 512, 'p', 512, 512, 512, 512, 'p'],
}

class VGG:

    def __init__(self, x, image_size, image_channels, num_classes, phase_train, net):
        self.x_image = tf.reshape(x, [-1, image_size, image_size, image_channels])
        self.num_classes = num_classes
        self.phase_train = phase_train
        self.net = net

    def build_network(self):
        inputs = self.x_image
        net = nets[self.net]
        for num_filters in net:
            if num_filters == 'p':
                out = tf.layers.max_pooling2d(
                                       inputs, 
                                       pool_size=[2, 2], 
                                       strides=2, 
                                       padding='SAME'
                                       )
            else:
                out = tf.layers.conv2d(
                                inputs=inputs,
                                filters=num_filters,
                                kernel_size=[3, 3],
                                padding='SAME',
                                activation=None
                                )
                out = self.batch_norm(out, num_filters, self.phase_train)
                out = tf.nn.relu(out)
            inputs = out
        # flatten
        flat = tf.reshape(out, [-1, int(out.shape[1]*out.shape[2]*out.shape[3])])
        # fc1
        fc1 = tf.layers.dense(inputs=flat, units=512, activation=None)
        bn = self.batch_norm(fc1, 512, self.phase_train)
        relu = tf.nn.relu(bn)
        drop = tf.layers.dropout(relu, rate=0.5)
        #fc2
        fc2 = tf.layers.dense(inputs=drop, units=512, activation=None)
        bn = self.batch_norm(fc2, 512, self.phase_train)
        relu = tf.nn.relu(bn)
        drop = tf.layers.dropout(fc2, rate=0.5)
        #softmax
        outputs = tf.layers.dense(inputs=drop, units=self.num_classes, activation=tf.nn.softmax)
        return outputs

    def batch_norm(self, x, n_out, phase_train):
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