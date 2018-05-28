import tensorflow as tf

class GoogLeNet:

    def __init__(self, x, image_size, image_channels, num_classes, phase_train, net):
        self.x_image = tf.reshape(x, [-1, image_size, image_size, image_channels])
        self.num_classes = num_classes
        self.phase_train = phase_train
        self.net = net

    def inception(self, inputs, 
                  num_filters_1_conv1, 
                  num_filters_2_conv1, num_filters_2_conv3, 
                  num_filters_3_conv1, num_filters_3_conv5, 
                  num_filters_4_conv1
                  ):
        '''
        branch1: inputs -> conv1x1 -> out1
        branch2: inputs -> conv1x1-> conv3x3 -> out2
        branch3: inputs -> conv1x1-> conv5x5 -> out3
        branch4: inputs -> maxpool3x3-> conv1x1 -> out34
        output: concat(out1, out2, out3, out4)
        '''
        out1 = tf.layers.conv2d(
                                inputs=inputs,
                                filters=num_filters_1_conv1,
                                kernel_size=[1, 1],
                                padding='SAME',
                                activation=None
                                )
        out1 = self.batch_normalization_layer(out1, num_filters_1_conv1, self.phase_train)
        out1 = tf.nn.relu(out1)

        out2 = tf.layers.conv2d(
                                inputs=inputs,
                                filters=num_filters_2_conv1,
                                kernel_size=[1, 1],
                                padding='SAME',
                                activation=None
                                )
        out2 = self.batch_normalization_layer(out2, num_filters_2_conv1, self.phase_train)
        out2 = tf.nn.relu(out2)
        out2 = tf.layers.conv2d(
                                inputs=out2,
                                filters=num_filters_2_conv3,
                                kernel_size=[3, 3],
                                padding='SAME',
                                activation=None
                                )
        out2 = self.batch_normalization_layer(out2, num_filters_2_conv3, self.phase_train)
        out2 = tf.nn.relu(out2)

        out3 = tf.layers.conv2d(
                                inputs=inputs,
                                filters=num_filters_3_conv1,
                                kernel_size=[1, 1],
                                padding='SAME',
                                activation=None
                                )
        out3 = self.batch_normalization_layer(out3, num_filters_3_conv1, self.phase_train)
        out3 = tf.nn.relu(out3)
        out3 = tf.layers.conv2d(
                                inputs=out2,
                                filters=num_filters_3_conv5,
                                kernel_size=[5, 5],
                                padding='SAME',
                                activation=None
                                )
        out3 = self.batch_normalization_layer(out3, num_filters_3_conv5, self.phase_train)
        out3 = tf.nn.relu(out3)

        out4 = tf.layers.max_pooling2d(
                                       inputs, 
                                       pool_size=[3, 3], 
                                       strides=1, 
                                       padding='SAME'
                                        )
        out4 = tf.layers.conv2d(
                                inputs=out4,
                                filters=num_filters_4_conv1,
                                kernel_size=[1, 1],
                                padding='SAME',
                                activation=None
                                )
        out4 = self.batch_normalization_layer(out4, num_filters_4_conv1, self.phase_train)
        out4 = tf.nn.relu(out4)

        return tf.concat([out1, out2, out3, out4], 3)

    def build_network(self):
        inputs = self.x_image
        # stem network
        prev1 = tf.layers.conv2d(inputs, filters=64, kernel_size=[3, 3], padding='SAME', activation=None)
        prev1 = self.batch_normalization_layer(prev1, 64, self.phase_train)
        prev1 = tf.nn.relu(prev1)
        prev1 = tf.layers.max_pooling2d(prev1, pool_size=[3, 3], strides=2, padding='SAME')
        prev2 = tf.layers.conv2d(prev1, filters=128, kernel_size=[1, 1], padding='SAME', activation=None)
        prev2 = self.batch_normalization_layer(prev2, 128, self.phase_train)
        prev2 = tf.nn.relu(prev2)
        prev2 = tf.layers.conv2d(prev2, filters=192, kernel_size=[3, 3], padding='SAME', activation=None)
        prev2 = self.batch_normalization_layer(prev2, 192, self.phase_train)
        prev2 = tf.nn.relu(prev2)
        prev2 = tf.layers.max_pooling2d(prev2, pool_size=[3, 3], strides=2, padding='SAME')
        
        # Stacked Inception Modules
        inception1 = self.inception(prev2, 64, 96, 128, 16, 32, 32)
        inception2 = self.inception(inception1, 128, 128, 192, 32, 96, 64)

        inception2 = tf.layers.max_pooling2d(inception2, pool_size=[3, 3], strides=2, padding='SAME')
        
        inception3 = self.inception(inception2, 192,  96, 208, 16,  48,  64)
        inception4 = self.inception(inception3, 160, 112, 224, 24,  64,  64)
        inception5 = self.inception(inception4, 128, 128, 256, 24,  64,  64)
        inception6 = self.inception(inception5, 112, 144, 288, 32,  64,  64)
        inception7 = self.inception(inception6, 256, 160, 320, 32, 128, 128)
        inception8 = self.inception(inception7, 256, 160, 320, 32, 128, 128)
        inception9 = self.inception(inception8, 384, 192, 384, 48, 128, 128)

        out = tf.layers.average_pooling2d(inception9, pool_size=[8, 8], strides=1, padding='SAME')

        # flatten
        flat = tf.reshape(out, [-1, int(out.shape[1]*out.shape[2]*out.shape[3])])

        #softmax
        outputs = tf.layers.dense(inputs=flat, units=self.num_classes, activation=tf.nn.softmax)
        return outputs

    def batch_normalization_layer(self, x, n_out, phase_train):
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