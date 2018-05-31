import numpy as np
import tensorflow as tf
import time
import argparse
from data import *
from model.vgg import VGG
from model.googlenet import GoogLeNet
from model.resnet import ResNet
from sklearn.model_selection import KFold #cross validation

IMAGE_SIZE = 32
IMAGE_CHANNEL = 3
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCH = 100
SAVE_PATH = "./checkpoint/"
SPLIT_SIZE = 10


def train():
    # data
    train_images, train_labels = get_train_batch()
    test_images, test_labels = get_test_batch()

    # model
    sess = tf.InteractiveSession()

    # variables
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, train_images.shape[1]], name='x_input')
        y = tf.placeholder(tf.float32, shape=[None, train_labels.shape[1]], name='y_input')

    # network
    with tf.name_scope('network'):
        #outputs = GoogLeNet(x, IMAGE_SIZE, IMAGE_CHANNEL, NUM_CLASSES, phase_train, '')
        outputs = ResNet(x, IMAGE_SIZE, IMAGE_CHANNEL, NUM_CLASSES, phase_train, 'res50')

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y))
    tf.summary.scalar('loss', loss)

    with tf.name_scope('accuracy'):
        predict = tf.argmax(outputs, axis=1)
        acc = tf.equal(predict, tf.argmax(y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(acc, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('train'):
        optimizer = tf.train.MomentumOptimizer(learning_rate=args.lr, momentum=0.9).minimize(loss, global_step=global_step)

    # saver
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(SAVE_PATH, sess.graph)

    if args.ckpt is not None:
        print("Trying to restore from checkpoint ...")
        try:
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=args.ckpt)
            saver.restore(sess, save_path=last_chk_path)
        except ValueError:
            print("Failed to restore checkpoint. Initializing variables instead.")
            sess.run(tf.global_variables_initializer())
    else:
        sess.run(tf.global_variables_initializer())


    global_val_acc = 0
    for train_idx, val_idx in KFold(n_splits=SPLIT_SIZE).split(train_images, train_labels):
        train_x = train_images[train_idx]
        train_y = train_labels[train_idx]
        val_x = train_images[val_idx]
        val_y = train_labels[val_idx]
        for e in range(EPOCH):
            for batch_index in range(0, len(train_x), BATCH_SIZE):
                if batch_index + BATCH_SIZE < len(train_x):
                    data = train_x[batch_index:batch_index+BATCH_SIZE]
                    label = train_y[batch_index:batch_index + BATCH_SIZE]
                else:
                    data = train_x[batch_index:len(train_images)]
                    label = train_y[batch_index:len(train_labels)]
                start_time = time.time()
                step, _, batch_loss = sess.run(
                                                          [global_step, optimizer, loss],
                                                          feed_dict={x: data, y: label, phase_train: True}
                                                          )
                end_time = time.time()
                duration = end_time - start_time
                # progress bar
                if batch_index % 20 == 0:
                    percentage = float(batch_index+BATCH_SIZE+e*len(train_x))/float(len(train_x)*EPOCH)*100.
                    bar_len = 29
                    filled_len = int((bar_len*int(percentage))/100)
                    bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len)
                    msg = "Epoch: {:}/{:} - Step: {:>5} - [{}] {:.2f}% - Loss: {:.4f} - {:} Sample/sec"
                    print(msg.format((e+1), EPOCH, step, bar, percentage, batch_loss, int(BATCH_SIZE/duration)))

            summary = tf.Summary(value=[
                                        tf.Summary.Value(tag='Epoch', simple_value=e),
                                        tf.Summary.Value(tag='Loss', simple_value=batch_loss)
                                        ]
                                )
            train_writer.add_summary(summary, step)

            # validation
            predicted_matrix = np.zeros(shape=len(val_x), dtype=np.int)
            for batch_index in range(0, len(val_x), BATCH_SIZE):
                if batch_index + BATCH_SIZE < len(val_x):
                    data = val_x[batch_index:batch_index+BATCH_SIZE]
                    label = val_y[batch_index:batch_index + BATCH_SIZE]
                    predicted_matrix[batch_index:batch_index+BATCH_SIZE] = sess.run(predict, feed_dict={x: data, y: label, phase_train: False})
                else:
                    data = val_x[batch_index:len(val_x)]
                    label = val_y[batch_index:len(val_y)]
                    predicted_matrix[batch_index:len(val_y)] = sess.run(predict, feed_dict={x: data, y: label, phase_train: False})
            correct = (np.argmax(val_y, axis=1) == predicted_matrix)
            acc = correct.mean()*100
            correct_numbers = correct.sum()
            mes = "\nValidation Accuracy: {:.2f}% ({}/{})\n"
            print(mes.format(acc, correct_numbers, len(val_y)))
            
            if acc > global_val_acc:
                saver.save(sess, SAVE_PATH+str(e)+'_'+str(args.lr)+'_acc:'+str(acc)+'.ckpt')
                global_test_acc = acc
                print("\nReach a better validation accuracy at epoch: {:} with {:.2f}%".format(e, acc))
                print("Saving at ... %s" % SAVE_PATH+str(EPOCH)+'_'+str(args.lr)+'.ckpt\n')
    train_writer.close()

    # test section
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
    mes = "\nTest Accuracy: {:.2f}% ({}/{})"
    print(mes.format(acc, correct_numbers, len(test_labels)))

    saver.save(sess, SAVE_PATH+str(e)+'_'+str(args.lr)+'_acc:'+str(acc)+'.ckpt')
    global_test_acc = acc
    print("\nReach a better testing accuracy at epoch: {:} with {:.2f}%".format(e, acc))
    print("Saving at ... %s" % SAVE_PATH+str(EPOCH)+'_'+str(args.lr)+'.ckpt')    

def main():
    train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--ckpt', type=str, help='checkpoint path')
    args = parser.parse_args()
    main()


sess.close()


