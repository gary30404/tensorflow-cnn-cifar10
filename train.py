import numpy as np
import tensorflow as tf
import time

from data import *
from model import *

BATCH_SIZE = 128
EPOCH = 60
SAVE_PATH = "./tensorboard/cifar-10-v1.0.0/"

# data
train_images, train_labels = get_train_batch()
test_images, test_label = get_test_batch()

# variables
x = tf.placeholder(tf.float32, shape=[None, train_images.shape[1]], name='input')
y = tf.placeholder(tf.float32, shape=[None, train_labels.shape[1]], name='output')
phase_train = tf.placeholder(tf.bool, name='phase_train')
global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')

# network
loss, outputs, predict, accuracy = network(x, y, phase_train)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-08
                                   ).minimize(loss, global_step=global_step)

# saver
merged = tf.summary.merge_all()
saver = tf.train.Saver()
sess = tf.Session()
train_writer = tf.summary.FileWriter(SAVE_PATH, sess.graph)

sess.run(tf.global_variables_initializer())


def train():
    tacc = 0
    for e in range(EPOCH):
        for batch_index in range(0, len(train_images), BATCH_SIZE):
            if batch_index + BATCH_SIZE < len(train_images):
                data = train_images[batch_index:batch_index+BATCH_SIZE]
                label = train_labels[batch_index:batch_index + BATCH_SIZE]
            else:
                data = train_images[batch_index:len(train_images)]
                label = train_labels[batch_index:len(train_labels)]
            start_time = time.time()
            step, _, batch_loss, batch_acc = sess.run(
                                                      [global_step, optimizer, loss, accuracy],
                                                      feed_dict={x: data, y: label, phase_train: True}
                                                      )
            end_time = time.time()
            duration = end_time - start_time
            tacc += round(batch_acc*BATCH_SIZE)
            # progress bar
            if batch_index % 20 == 0:
                percentage = int(round((((batch_index+BATCH_SIZE+e*len(train_images))/(len(train_images)*EPOCH))*100)))
                bar_len = 29
                filled_len = int((bar_len*int(percentage))/100)
                bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len)
                msg = "Epoch: {:}/{:} - Step: {:>5} - [{}] {:>3}% - Bacc: {:.2f} - Tacc: {:.2f} - loss: {:.4f} - {:} sample/sec"
                print(msg.format((e+1), EPOCH, step, bar, percentage, batch_acc, tacc/(batch_index+BATCH_SIZE+e*len(train_images)), batch_loss, int(BATCH_SIZE/duration)))

        summary = tf.Summary(value=[
                                    tf.Summary.Value(tag='Epoch', simple_value=e),
                                    tf.Summary.Value(tag='Loss', simple_value=batch_loss),
                                    tf.Summary.Value(tag='Tacc', simple_value=tacc/(batch_index+BATCH_SIZE+e*len(train_images))),
                                    ]
                            )
        train_writer.add_summary(summary, step)
        saver.save(sess, save_path=SAVE_PATH, global_step=step)
        test()


def test():

    predicted_matrix = np.zeros(shape=len(test_images), dtype=np.int)
    for batch_index in range(0, len(test_images), BATCH_SIZE):
        if batch_index + BATCH_SIZE < len(test_images):
            data = test_images[batch_index:batch_index+BATCH_SIZE]
            label = test_labels[batch_index:batch_index + BATCH_SIZE]
            predicted_matrix[batch_index:batch_index+BATCH_SIZE] = sess.run(predict, feed_dict={x: data, y: label, phase_train: False})
        else:
            data = test_images[batch_index:len(test_images)]
            label = test_labels[batch_index:len(test_labels)]
            predicted_matrix[batch_index:len(test_labels)] = sess.run(predict, feed_dict={x: data, y: label, phase_train: False})
    correct = (np.argmax(test_labels, axis=1) == predicted_matrix)
    acc = correct.mean()*100
    correct_numbers = correct.sum()
    mes = "\nAccuracy: {:.2f}% ({}/{})"
    print(mes.format(acc, correct_numbers, len(test_labels)))


def main():
    train()


if __name__ == "__main__":
    main()


sess.close()