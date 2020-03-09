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
from lib.dataset import input_fn
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


def train_and_eval(model):
    for n in range(FLAGS.train_epochs):
        tf.compat.v1.logging.info('=' * 30 + ' START EPOCH {} '.format(n + 1) + '=' * 30 + '\n')
        train_data_list = list_files(FLAGS.train_data)  # dir to file list
        for f in train_data_list:
            t0 = time.time()
            tf.compat.v1.logging.info('<EPOCH {}>: Start training {}'.format(n + 1, f))
            model.train(
                input_fn=lambda: input_fn(f, FLAGS.image_train_data, 'train', FLAGS.batch_size),
                hooks=None,
                steps=None,
                max_steps=None,
                saving_listeners=None)
            tf.compat.v1.logging.info('<EPOCH {}>: Finish training {}, take {} mins'.format(n + 1, f, elapse_time(t0)))
            print('-' * 80)
            tf.compat.v1.logging.info('<EPOCH {}>: Start evaluating {}'.format(n + 1, FLAGS.eval_data))
            t0 = time.time()
            results = model.evaluate(
                input_fn=lambda: input_fn(FLAGS.eval_data, FLAGS.image_eval_data, 'eval', FLAGS.batch_size),
                steps=None,  # Number of steps for which to evaluate model.
                hooks=None,
                checkpoint_path=None,  # latest checkpoint in model_dir is used.
                name=None)
            tf.compat.v1.logging.info('<EPOCH {}>: Finish evaluation {}, take {} mins'.format(n + 1, FLAGS.eval_data, elapse_time(t0)))
            print('-' * 80)
            # Display evaluation metrics
            for key in sorted(results):
                print('{}: {}'.format(key, results[key]))
        # every epochs_per_eval test the model (use larger test dataset)
        if (n+1) % FLAGS.epochs_per_eval == 0:
            tf.compat.v1.logging.info('<EPOCH {}>: Start testing {}'.format(n + 1, FLAGS.test_data))
            results = model.evaluate(
                input_fn=lambda: input_fn(FLAGS.test_data, FLAGS.image_test_data, 'pred', FLAGS.batch_size),
                 steps=None,  # Number of steps for which to evaluate model.
                 hooks=None,
                 checkpoint_path=None,  # If None, the latest checkpoint in model_dir is used.
                 name=None)
            tf.compat.v1.logging.info('<EPOCH {}>: Finish testing {}, take {} mins'.format(n + 1, FLAGS.test_data, elapse_time(t0)))
            print('-' * 80)
            # Display evaluation metrics
            for key in sorted(results):
                print('{}: {}'.format(key, results[key]))


def dynamic_train(model):
    """Dynamic train mode.
    For example:
        train_data_files: [0301, 0302, 0303, ...]
        train mode:
            first take 0301 as train data, 0302 as test data;
            then keep training take 0302 as train data, 0303 as test data ...
    """
    data_files = list_files(FLAGS.train_data)
    data_files.sort()
    assert len(data_files) > 1, 'Dynamic train mode need more than 1 data file'

    for i in range(len(data_files)-1):
        train_data = data_files[i]
        test_data = data_files[i+1]
        tf.compat.v1.logging.info('=' * 30 + ' START TRAINING DATA: {} '.format(train_data) + '=' * 30 + '\n')
        for n in range(FLAGS.train_epochs):
            t0 = time.time()
            tf.compat.v1.logging.info('START TRAIN DATA <{}> <EPOCH {}>'.format(train_data, n + 1))
            model.train(
                input_fn=lambda: input_fn(train_data, FLAGS.image_train_data, 'train', FLAGS.batch_size),
                hooks=None,
                steps=None,
                max_steps=None,
                saving_listeners=None)
            tf.compat.v1.logging.info('FINISH TRAIN DATA <{}> <EPOCH {}> take {} mins'.format(train_data, n + 1, elapse_time(t0)))
            print('-' * 80)
            tf.compat.v1.logging.info('START EVALUATE TEST DATA <{}> <EPOCH {}>'.format(test_data, n + 1))
            t0 = time.time()
            results = model.evaluate(
                input_fn=lambda: input_fn(test_data, FLAGS.image_eval_data, 'eval', FLAGS.batch_size),
                steps=None,  # Number of steps for which to evaluate model.
                hooks=None,
                checkpoint_path=None,  # latest checkpoint in model_dir is used.
                name=None)
            tf.compat.v1.logging.info('FINISH EVALUATE TEST DATA <{}> <EPOCH {}>: take {} mins'.format(test_data, n + 1, elapse_time(t0)))
            print('-' * 80)
            # Display evaluation metrics
            for key in sorted(results):
                print('{}: {}'.format(key, results[key]))


def train(model):
    for n in range(FLAGS.train_epochs):
        tf.compat.v1.logging.info('=' * 30 + ' START EPOCH {} '.format(n + 1) + '=' * 30 + '\n')
        train_data_list = list_files(FLAGS.train_data)  # dir to file list
        for f in train_data_list:
            t0 = time.time()
            tf.compat.v1.logging.info('<EPOCH {}>: Start training {}'.format(n + 1, f))
            model.train(
                input_fn=lambda: input_fn(f, FLAGS.image_train_data, 'train', FLAGS.batch_size),
                hooks=None,
                steps=None,
                max_steps=None,
                saving_listeners=None)
            tf.compat.v1.logging.info('<EPOCH {}>: Finish training {}, take {} mins'.format(n + 1, f, elapse_time(t0)))


def train_and_eval_api(model):
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(FLAGS.train_data, FLAGS.image_train_data, FLAGS.batch_size), max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(FLAGS.eval_data, FLAGS.image_eval_data, FLAGS.batch_size))
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


def main(unused_argv):
    CONFIG = Config()
    print("Using TensorFlow Version %s" % tf.__version__)
    # assert "1.4" <= tf.__version__, "Need TensorFlow r1.4 or Later."
    print('\nModel Type: {}'.format(FLAGS.model_type))
    model_dir = os.path.join(FLAGS.model_dir, FLAGS.model_type)
    print('\nModel Directory: {}'.format(model_dir))

    print("\nUsing Train Config:")
    for k, v in CONFIG.train.items():
        print('{}: {}'.format(k, v))
    print("\nUsing Model Config:")
    for k, v in CONFIG.model.items():
        print('{}: {}'.format(k, v))

    if not FLAGS.keep_train:
        # Clean up the model directory if not keep training
        shutil.rmtree(model_dir, ignore_errors=True)
        print('Remove model directory: {}'.format(model_dir))
    # model = build_estimator(model_dir, FLAGS.model_type)
    model = build_custom_estimator(model_dir, FLAGS.model_type)
    tf.compat.v1.logging.info('Build estimator: {}'.format(model))

    if CONFIG.train['dynamic_train']:
        train_fn = dynamic_train
        print("Using dynamic train mode.")
    else:
        train_fn = train_and_eval

    if CONFIG.distribution["is_distribution"]:
        print("Using PID: {}".format(os.getpid()))
        cluster = CONFIG.distribution["cluster"]
        job_name = CONFIG.distribution["job_name"]
        task_index = CONFIG.distribution["task_index"]
        print("Using Distributed TensorFlow. Local host: {} Job_name: {} Task_index: {}"
              .format(cluster[job_name][task_index], job_name, task_index))
        cluster = tf.train.ClusterSpec(CONFIG.distribution["cluster"])
        server = tf.distribute.Server(cluster,
                                 job_name=job_name,
                                 task_index=task_index)
        # distributed can not including eval.
        train_fn = train
        if job_name == 'ps':
            # wait for incoming connection forever
            server.join()
            # sess = tf.Session(server.target)
            # queue = create_done_queue(task_index, num_workers)
            # for i in range(num_workers):
            #     sess.run(queue.dequeue())
            #     print("ps {} received worker {} done".format(task_index, i)
            # print("ps {} quitting".format(task_index))
        else:  # TODO：supervisor & MonotoredTrainingSession & experiment (deprecated)
            train_fn(model)
            # train_and_eval(model)
            # Each worker only needs to contact the PS task(s) and the local worker task.
            # config = tf.ConfigProto(device_filters=[
            #     '/job:ps', '/job:worker/task:%d' % arguments.task_index])
            # with tf.device(tf.train.replica_device_setter(
            #         worker_device="/job:worker/task:%d" % task_index,
            #         cluster=cluster)):
            # e = _create_experiment_fn()
            # e.train_and_evaluate()  # call estimator's train() and evaluate() method
            # hooks = [tf.train.StopAtStepHook(last_step=10000)]
            # with tf.train.MonitoredTrainingSession(
            #         master=server.target,
            #         is_chief=(task_index == 0),
            #         checkpoint_dir=args.model_dir,
            #         hooks=hooks) as mon_sess:
            #     while not mon_sess.should_stop():
            #         # mon_sess.run()
            #         classifier.fit(input_fn=train_input_fn, steps=1)
    else:
        # local run
        train_fn(model)


if __name__ == '__main__':
    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
