#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/15
"""Training Wide and Deep Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import shutil
import sys
import time

import tensorflow as tf

from lib.read_conf import Config
from lib.dataset import input_fn, input_fn_show
from lib.build_estimator import build_estimator, build_custom_estimator
from lib.utils.util import elapse_time, list_files

CONFIG = Config().train
parser = argparse.ArgumentParser(description='Train Wide and Deep Model.')

parser.add_argument(
    '--model_dir', type=str, default=CONFIG["model_dir"],
    help='Base directory for the model.')
parser.add_argument(
    '--model_type', type=str, default=CONFIG["model_type"],
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")
parser.add_argument(
    '--train_epochs', type=int, default=CONFIG["train_epochs"],
    help='Number of training epochs.')
parser.add_argument(
    '--epochs_per_eval', type=int, default=CONFIG["epochs_per_eval"],
    help='The number of training epochs to run between evaluations.')
parser.add_argument(
    '--batch_size', type=int, default=CONFIG["batch_size"],
    help='Number of examples per batch.')
parser.add_argument(
    '--train_data', type=str, default=CONFIG["train_data"],
    help='Path to the train data.')
parser.add_argument(
    '--eval_data', type=str, default=CONFIG["eval_data"],
    help='Path to the validation data.')
parser.add_argument(
    '--pred_data', type=str, default=CONFIG["pred_data"],
    help='Path to the validation data.')
parser.add_argument(
    '--test_data', type=str, default=CONFIG["test_data"],
    help='Path to the test data.')
parser.add_argument(
    '--image_train_data', type=str, default=CONFIG["image_train_data"],
    help='Path to the train data.')
parser.add_argument(
    '--image_eval_data', type=str, default=CONFIG["image_eval_data"],
    help='Path to the train data.')
parser.add_argument(
    '--image_test_data', type=str, default=CONFIG["image_test_data"],
    help='Path to the train data.')
parser.add_argument(
    '--keep_train', type=int, default=CONFIG["keep_train"],
    help='Whether to keep training on previous trained model.')


# def input_fn(self, data_file, mode, batch_size):
#     assert mode in {'train', 'eval', 'pred'}, (
#         'mode must in `train`, `eval`, or `pred`, found {}'.format(mode))
#     tf.logging.info('Parsing input csv files: {}'.format(data_file))
#     # Extract lines from input files using the Dataset API.
#     dataset = tf.data.TextLineDataset(data_file)
#     # Use `Dataset.map()` to build a pair of a feature dictionary
#     # and a label tensor for each example.
#     # Shuffle, repeat, and batch the examples.
#     dataset = dataset.map(
#         self._parse_csv(is_pred=(mode == 'pred')),
#         num_parallel_calls=self._num_parallel_calls)
#     if mode == 'train':
#         dataset = dataset.shuffle(buffer_size=self._shuffle_buffer_size, seed=123)
#
#     dataset = dataset.prefetch(2 * batch_size)
#     if self._multivalue:
#         padding_dic = {k: [None] for k in self._feature_used}
#         if self._use_weight:
#             padding_dic['weight_column'] = [None]
#         padded_shapes = padding_dic if mode == 'pred' else (padding_dic, [None])
#         dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
#     else:
#         # batch(): each element tensor must have exactly same shape, change rank 0 to rank 1
#         dataset = dataset.batch(batch_size)
#     return dataset.make_one_shot_iterator().get_next()

if __name__ == '__main__':
    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    parsed_input = input_fn_show(csv_data_file=FLAGS.pred_data, img_data_file=None,
             mode='pred', batch_size=FLAGS.batch_size)
    # print(parsed_input)

    # with tf.Session() as sess:
    #     tf.print(parsed_input['ucomp'], output_stream=sys.stdout)
    # for line in parsed_input['ucomp'].take(5):
    #     print(line.numpy())
    #
    # with tf.Session() as sess:
    #     for column in parsed_input:
    #         parsed_input[column].eval()
    #         print(parsed_input[column])
