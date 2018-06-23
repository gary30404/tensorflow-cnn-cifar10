import pickle
import numpy as np
import os
from urllib import urlretrieve
import tarfile
import sys
import time
from augment import augment

DATASET_DIRECTORY = './dataset/'
CIFAR10_DIRECTORY = 'cifar-10-batches-py/'
URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
META_FILE = DATASET_DIRECTORY+CIFAR10_DIRECTORY+'batches.meta'
NUM_CLASSES = 10

def get_data(file):

    f = open(file, 'rb')
    data_dict = pickle.load(f)
    f.close()

    X = np.array(data_dict['data'], dtype=float)
    Y = np.eye(NUM_CLASSES)[np.array(data_dict['labels'])]

    X /= 255.0
    X = X.reshape([-1, 3, 32, 32])
    X = X.transpose([0, 2, 3, 1])
    X -= (0.4914, 0.4822, 0.4465)
    X /= (0.2023, 0.1994, 0.2010)
    X = X.reshape(-1, 32*32*3)

    return X,Y

def get_train_batch():
    maybe_download_and_extract()
    f = open(META_FILE, 'rb')
    f.close()
    BATCH_FILE = DATASET_DIRECTORY+CIFAR10_DIRECTORY+'/data_batch_'
    x, y = get_data(BATCH_FILE+'1')
    for i in range(4):
        xx, yy = get_data(BATCH_FILE+str(i+2))
        x = np.concatenate((x ,xx), axis=0)
        y = np.concatenate((y, yy), axis = 0)
    return x, y

def get_test_batch():
    maybe_download_and_extract()
    f = open(META_FILE, 'rb')
    f.close()
    BATCH_FILE = DATASET_DIRECTORY+CIFAR10_DIRECTORY+'/test_batch'
    x, y = get_data(BATCH_FILE)
    return x, y


def print_download_progress(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = count * block_size
    pct_complete = float(progress_size) / total_size
    speed = int(progress_size / (1024 * duration))
    msg = "\r- Download progress: {0:.1%}, {1:} MB, {2:} KB/s, {3:} seconds passed".format(pct_complete, progress_size/(1024*1024), speed, int(duration))
    sys.stdout.write(msg)
    sys.stdout.flush()


def maybe_download_and_extract():
    if not os.path.exists(DATASET_DIRECTORY):
        os.makedirs(DATASET_DIRECTORY)
        filename = URL.split('/')[-1]
        file_path = os.path.join(DATASET_DIRECTORY, filename)
        file_path, _ = urlretrieve(url=URL, filename=file_path, reporthook=print_download_progress)
        print("\nDownload finished. Extracting files.")
        tarfile.open(name=file_path, mode="r:gz").extractall(DATASET_DIRECTORY)
        print("Done.")
