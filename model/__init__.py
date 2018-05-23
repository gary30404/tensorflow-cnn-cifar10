import tensorflow as tf
from vgg import VGG

def network(x, y, image_size, image_channels, num_classes, phase_train):

    net = VGG(x, image_size, image_channels, num_classes, phase_train, 'VGG16')
    outputs = net.build_network()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y))
    predict = tf.argmax(outputs, axis=1)
    acc = tf.equal(predict, tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(acc, tf.float32))

    return loss, outputs, predict, accuracy

