import pandas as pd
import numpy as np
import argparse
import pandas as pd
import operator
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import itemfreq


import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Activation, concatenate
# from tensorflow.keras.layers.advanced_activations import ReLU
# from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model

from lib.read_conf import Config
from lib.dataset import input_fn

# fix ImportError: No mudule named lib.*

CONF = Config()
CONFIG = CONF

def get_dataset():
    feature_all = CONF.get_feature_name()  # all features
    feature_conf = CONF.read_feature_conf()  # feature conf dict
    feature_type = {}

    hashed_cat_feature = []
    hash_bucket_size = {}
    for f in feature_all:
        if f in feature_conf:  # used features
            conf = feature_conf[f]
            if conf['type'] == 'category' and conf["transform"] == 'hash_bucket':
                hashed_cat_feature.append(f)
                hash_bucket_size[f] = conf["parameter"]
        else:  # unused features
            feature_type[f] = 'str'

        li = []
        for filename in ["../data/train/train1", "../data/train/train2", "../data/eval/eval1", "../data/test/test1"]:
            df = pd.read_table(filename, names=['label'] + feature_all, dtype=str)
            x, y = df[feature_all].copy(), df['label'].copy()
            li.append(x)

        for filename in ["../data/pred/pred1"]:
            x = pd.read_table(filename, names=feature_all, dtype=str)
            li.append(x)
        x = pd.concat(li, axis=0, ignore_index=True)
        # table = pd.read_table("../data/train/train1", names=['label'] + feature_all, dtype=str)
        # x, y = table[feature_all].copy(), table['label'].copy()


        # drop unused
        for col in x.columns:
            if col not in hashed_cat_feature:
                x.pop(col)
        x_1, y_1 = x.iloc[0:5], y.iloc[0:5]

    return x, hash_bucket_size


if __name__ == "__main__":

    dataset, hash_bucket_size = get_dataset()

    # frequency before hashing
    # raw_freq_list = np.array([])
    # for col_name in dataset:
    #     col = dataset[col_name]
    #     freq = itemfreq(col)
    #     len_freq = len(freq)
    #     raw_freq_sorted = np.flip(freq[np.argsort(freq[:, 1])], axis=0)  # (bucket_id, number of access)
    #     raw_freq_list = np.concatenate((raw_freq_list, raw_freq_sorted[:, 1]))


    hash_id = []
    for col in dataset.columns:
        print("bucket size: {}".format(hash_bucket_size[col]))
        hash_id.append(tf.strings.to_hash_bucket(
        list(dataset[col]), num_buckets=hash_bucket_size[col], name=None
    ))

    # string = ["asdfg", "daf"]
    # hash_id = tf.strings.to_hash_bucket(
    #     string, num_buckets=100, name=None
    # )

    with tf.Session() as sess:
        hash_id_result = sess.run(hash_id)

    # frequency after hashing
    freq_list = np.array([])
    for col in hash_id_result:
        freq = itemfreq(col)
        len_freq = len(freq)
        freq_sorted = np.flip(freq[np.argsort(freq[:, 1])], axis=0) # (bucket_id, number of access)
        freq_list = np.concatenate((freq_list, freq_sorted[:, 1]))

    print("length: ", len(freq_list))
    # given the data provided, total number of embedding that will be accessed is less than 1MB cache size
    freq_list_sorted = sorted(freq_list, reverse=True)
    dim = 8
    float_size = 4
    # BRAM cache = 1MB
    cache_entry_num = 1024 * 1024 / dim / float_size
    print("Entry number of 1MB BRAM: {}".format(cache_entry_num))
    print("Total number of accessed entry in the dataset: {}".format(len(freq_list_sorted)))

    total_access = np.sum(freq_list_sorted)
    x_axis = np.arange(1, total_access+ 1)
    freq_cumulative_sum = np.cumsum(freq_list_sorted)
    y_axis = np.full(shape=x_axis.shape, fill_value=freq_cumulative_sum[-1])
    y_axis[:len(freq_cumulative_sum)] = freq_cumulative_sum
    x_axis /= total_access
    y_axis /= total_access
    plt.plot(x_axis, y_axis)
    plt.title('Embedding Frequency Distribution')
    # plt.ylabel('')
    # plt.xlabel('L (iteration)')
    # plt.xscale("log")
    plt.show()
    plt.save("../output/Embedding Frequency Distribution.png")

    for i in range(1, 21):
        perc_coverage = y_axis[int(i * 0.01 * total_access)]
        print("Top {}% of hot embedding covers {:.2f}% of accesses".format(i, perc_coverage * 100))
