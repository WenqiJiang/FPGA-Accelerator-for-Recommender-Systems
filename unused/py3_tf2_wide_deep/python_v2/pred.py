#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/2/2
"""Wide and Deep Model Prediction
Not support for custom classifier, cause use different variable name scope, key not found in checkpoint"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

from lib.read_conf import Config
from lib.dataset import input_fn
from lib.build_estimator import build_estimator, build_custom_estimator
from lib.utils.util import elapse_time

CONFIG = Config().train
parser = argparse.ArgumentParser(description='Wide and Deep Model Prediction')

parser.add_argument(
    '--model_dir', type=str, default=CONFIG["model_dir"],
    help='Model checkpoint dir for evaluating.')

parser.add_argument(
    '--model_type', type=str, default=CONFIG["model_type"],
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument(
    '--data_dir', type=str, default=CONFIG["pred_data"],
    help='Evaluating data dir.')

parser.add_argument(
    '--image_data_dir', type=str, default=None,
    help='Evaluating image data dir.')

parser.add_argument(
    '--batch_size', type=int, default=CONFIG["batch_size"],
    help='Number of examples per batch.')

parser.add_argument(
    '--checkpoint_path', type=str, default=CONFIG["checkpoint_path"],
    help="Path of a specific checkpoint to predict. If None, the latest checkpoint in model_dir is used.")


def main(unused_argv):
    print("Using TensorFlow version %s" % tf.__version__)
    # assert "1.4" <= tf.__version__, "TensorFlow r1.4 or later is needed"
    if FLAGS.data_dir is None:
        raise ValueError("Must specify prediction data_file by --data_dir")
    print('Model type: {}'.format(FLAGS.model_type))
    model_dir = os.path.join(FLAGS.model_dir, FLAGS.model_type)
    print('Model directory: {}'.format(model_dir))
    # model = build_estimator(model_dir, FLAGS.model_type)
    model = build_custom_estimator(model_dir, FLAGS.model_type)
    tf.compat.v1.logging.info('Build estimator: {}'.format(model))

    # weights and other parameters (e.g. Adagrad) of the model
    name_ls = model.get_variable_names()
    print_shape = True
    total_linear_weights = 0
    for name in name_ls:
        if print_shape:
            shape = model.get_variable_value(name).shape
            print(name, "\t", shape)
            if name[:6] == "linear" and \
                    (name[-7:] == "weights"or name[-4:] == "bias"):
                total_linear_weights += np.prod(shape)
        else:
            print(name)
    if print_shape:
        print("Total parameters in linear model: {}".format(total_linear_weights))

    # embedding layer look up
    sample_embedding = model.get_variable_value(
        'dnn/input_from_feature_columns/input_layer/ad_cates_embedding/embedding_weights')
    ids = [10, 20, 30]
    with tf.compat.v1.Session() as sess:
        lookup = tf.nn.embedding_lookup(params=sample_embedding,ids=ids).eval()
        print(lookup)

    # predictions
    tf.compat.v1.logging.info('='*30+'START PREDICTION'+'='*30)
    t0 = time.time()

    predictions = model.predict(input_fn=lambda: input_fn(FLAGS.data_dir, FLAGS.image_data_dir, 'pred', FLAGS.batch_size),
                                predict_keys=None,
                                hooks=None,
                                checkpoint_path=FLAGS.checkpoint_path)  # defaults None to use latest_checkpoint

    for pred_dict in predictions:  # dict{probabilities, classes, class_ids}
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print('\nPrediction is "{}" ({:.1f}%)'.format(class_id, 100 * probability))

    tf.compat.v1.logging.info('=' * 30 + 'FINISH PREDICTION, TAKE {} mins'.format(elapse_time(t0)) + '=' * 30)

if __name__ == '__main__':
    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
