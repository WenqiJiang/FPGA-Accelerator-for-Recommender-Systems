import pandas as pd
import numpy as np
import argparse
import pandas as pd
from datetime import datetime


import tensorflow as tf
from tensorflow import keras
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

parser = argparse.ArgumentParser(description='Train Wide and Deep Model.')

parser.add_argument(
    '--model_dir', type=str, default=CONFIG.train["model_dir"],
    help='Base directory for the model.')
parser.add_argument(
    '--model_type', type=str, default=CONFIG.train["model_type"],
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")
parser.add_argument(
    '--train_epochs', type=int, default=CONFIG.train["train_epochs"],
    help='Number of training epochs.')
parser.add_argument(
    '--epochs_per_eval', type=int, default=CONFIG.train["epochs_per_eval"],
    help='The number of training epochs to run between evaluations.')
# parser.add_argument(
#     '--batch_size', type=int, default=CONFIG.train["batch_size"],
#     help='Number of examples per batch.')
parser.add_argument(
    '--train_data', type=str, default=CONFIG.train["train_data"],
    help='Path to the train data.')
parser.add_argument(
    '--eval_data', type=str, default=CONFIG.train["eval_data"],
    help='Path to the validation data.')
parser.add_argument(
    '--pred_data', type=str, default=CONFIG.train["pred_data"],
    help='Path to the validation data.')
parser.add_argument(
    '--test_data', type=str, default=CONFIG.train["test_data"],
    help='Path to the test data.')
parser.add_argument(
    '--image_train_data', type=str, default=CONFIG.train["image_train_data"],
    help='Path to the train data.')
parser.add_argument(
    '--image_eval_data', type=str, default=CONFIG.train["image_eval_data"],
    help='Path to the train data.')
parser.add_argument(
    '--image_test_data', type=str, default=CONFIG.train["image_test_data"],
    help='Path to the train data.')
parser.add_argument(
    '--keep_train', type=int, default=CONFIG.train["keep_train"],
    help='Whether to keep training on previous trained model.')

# wide columns
categorical_column_with_identity = tf.feature_column.categorical_column_with_identity
categorical_column_with_hash_bucket = tf.feature_column.categorical_column_with_hash_bucket
categorical_column_with_vocabulary_list = tf.feature_column.categorical_column_with_vocabulary_list
crossed_column = tf.feature_column.crossed_column
bucketized_column = tf.feature_column.bucketized_column
# deep columns
embedding_column = tf.feature_column.embedding_column
indicator_column = tf.feature_column.indicator_column
numeric_column = tf.feature_column.numeric_column

def _build_model_columns():
    """
    Build wide and deep feature columns from custom feature conf using tf.feature_column API
    wide_columns: category features + cross_features + [discretized continuous features]
    deep_columns: continuous features + category features(onehot or embedding for sparse features) + [cross_features(embedding)]
    Return:
        _CategoricalColumn and __DenseColumn instance in tf.feature_column API
    """
    def embedding_dim(dim):
        """empirical embedding dim"""
        return int(np.power(2, np.ceil(np.log(dim**0.25))))

    def normalizer_fn_builder(scaler, normalization_params):
        """normalizer_fn builder"""
        if scaler == 'min_max':
            return lambda x: (x-normalization_params[0]) / (normalization_params[1]-normalization_params[0])
        elif scaler == 'standard':
            return lambda x: (x-normalization_params[0]) / normalization_params[1]
        else:
            return lambda x: tf.math.log(x)

    feature_conf_dic = CONF.read_feature_conf()
    cross_feature_list = CONF.read_cross_feature_conf()
    # tf.logging.info('Total used feature class: {}'.format(len(feature_conf_dic)))
    # tf.logging.info('Total used cross feature class: {}'.format(len(cross_feature_list)))

    wide_columns = []
    deep_columns = []
    wide_dim = 0
    deep_dim = 0
    feature_num = 0
    for feature, conf in feature_conf_dic.items():
        f_type, f_tran, f_param = conf["type"], conf["transform"], conf["parameter"]
        feature_num += 1
        if f_type == 'category':

            if f_tran == 'hash_bucket':
                hash_bucket_size = f_param
                embed_dim = embedding_dim(hash_bucket_size)
                col = categorical_column_with_hash_bucket(feature,
                    hash_bucket_size=hash_bucket_size,
                    dtype=tf.string)
                wide_columns.append(col)
                deep_columns.append(embedding_column(col,
                    dimension=embed_dim,
                    combiner='mean',
                    initializer=None,
                    ckpt_to_load_from=None,
                    tensor_name_in_ckpt=None,
                    max_norm=None,
                    trainable=True))
                wide_dim += hash_bucket_size
                deep_dim += embed_dim

            elif f_tran == 'vocab':
                col = categorical_column_with_vocabulary_list(feature,
                    vocabulary_list=list(map(str, f_param)),
                    dtype=None,
                    default_value=-1,
                    num_oov_buckets=0)  # len(vocab)+num_oov_buckets
                wide_columns.append(col)
                deep_columns.append(indicator_column(col))
                wide_dim += len(f_param)
                deep_dim += len(f_param)

            elif f_tran == 'identity':
                num_buckets = f_param
                col = categorical_column_with_identity(feature,
                    num_buckets=num_buckets,
                    default_value=0)  # Values outside range will result in default_value if specified, otherwise it will fail.
                wide_columns.append(col)
                deep_columns.append(indicator_column(col))
                wide_dim += num_buckets
                deep_dim += num_buckets
        else:
            normalizaton, boundaries = f_param["normalization"], f_param["boundaries"]
            if f_tran is None:
                normalizer_fn = None
            else:
                normalizer_fn = normalizer_fn_builder(f_tran, tuple(normalizaton))
            col = numeric_column(feature,
                 shape=(1,),
                 default_value=0,  # default None will fail if an example does not contain this column.
                 dtype=tf.float32,
                 normalizer_fn=normalizer_fn)
            if boundaries:  # whether include continuous features in wide part
                wide_columns.append(bucketized_column(col, boundaries=boundaries))
                wide_dim += (len(boundaries)+1)
            deep_columns.append(col)
            deep_dim += 1

    cross_feature_num = 0
    for cross_features, hash_bucket_size, is_deep in cross_feature_list:
        cf_list = []
        cross_feature_num += 1
        for f in cross_features:
            f_type = feature_conf_dic[f]["type"]
            f_tran = feature_conf_dic[f]["transform"]
            f_param = feature_conf_dic[f]["parameter"]
            if f_type == 'continuous':
                cf_list.append(bucketized_column(numeric_column(f, default_value=0), boundaries=f_param['boundaries']))
            else:
                if f_tran == 'identity':
                    # If an input feature is of numeric type, you can use categorical_column_with_identity
                    cf_list.append(categorical_column_with_identity(f, num_buckets=f_param,
                    default_value=0))
                else:
                    cf_list.append(f)  # category col put the name in crossed_column
        col = crossed_column(cf_list, hash_bucket_size)
        wide_columns.append(col)
        wide_dim += hash_bucket_size
        if is_deep:
            deep_columns.append(embedding_column(col, dimension=embedding_dim(hash_bucket_size)))
            deep_dim += embedding_dim(hash_bucket_size)

    print("feature_num: {}\ncross_feature_num: {}\ntotal_feature_num: {}".format(
        feature_num, cross_feature_num, feature_num + cross_feature_num))

    # add columns logging info
    tf.compat.v1.logging.info('Build total {} wide columns'.format(len(wide_columns)))
    for col in wide_columns:
        tf.compat.v1.logging.debug('Wide columns: {}'.format(col))
    tf.compat.v1.logging.info('Build total {} deep columns'.format(len(deep_columns)))
    for col in deep_columns:
        tf.compat.v1.logging.debug('Deep columns: {}'.format(col))
    tf.compat.v1.logging.info('Wide input dimension is: {}'.format(wide_dim))
    tf.compat.v1.logging.info('Deep input dimension is: {}'.format(deep_dim))

    return (wide_columns, wide_dim), (deep_columns, deep_dim)

class Wide_and_Deep:
    def __init__(self, mode='wide and deep'):
        self.mode = mode
        self.x_train_y_train = input_fn(
            csv_data_file="../data/train", img_data_file=None, mode="train", batch_size=1)
        self.x_test_y_test = input_fn(
            csv_data_file="../data/eval", img_data_file=None, mode="eval", batch_size=1)
        self.categ_inputs = None
        self.conti_input = None
        self.deep_component_outlayer = None
        self.logistic_input = None
        self.model = None

        (self.wide_columns, self.wide_dim), (self.deep_columns, self.deep_dim) = _build_model_columns()

    def deep_component(self):
        # feature_columns, e.g. hash columns (self.wide_columns / self.deep_columns)
        # https://www.tensorflow.org/tutorials/structured_data/feature_columns
        # tf.keras.layers.DenseFeatures:
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/layers/DenseFeatures
        self.input_layer = tf.keras.layers.DenseFeatures(feature_columns=self.deep_columns)
        # self.input_layer = tf.feature_column.input_layer(feature_columns=self.deep_columns)
        # concat_inputs = Activation('relu')(embedding_or_one_hot_feature)
        # bn_concat = BatchNormalization()(concat_inputs)
        fc1 = Dense(1024, use_bias=False)(self.input_layer)
        ac1 = ReLU()(fc1)
        bn1 = BatchNormalization()(ac1)
        fc2 = Dense(512, use_bias=False)(bn1)
        ac2 = ReLU()(fc2)
        bn2 = BatchNormalization()(ac2)
        fc3 = Dense(256)(bn2)
        ac3 = ReLU()(fc3)
        # WENQI
        deep_out = Dense(1)(ac3)
        self.deep_component_outlayer = deep_out

    def wide_component(self):
        self.wide_columns_processed = [tf.feature_column.indicator_column(w) for w in self.wide_columns]
        input_layer = tf.keras.layers.DenseFeatures(tf.feature_column.indicator_column(self.wide_columns_processed))
        # input_layer = tf.feature_column.input_layer(self.x_train,
        #                                             tf.feature_column.indicator_column(self.wide_columns))
        self.wide_component_outlayer = Dense(1)(input_layer)

    def load_model(self, filename='wide_and_deep.h5'):
        self.model = load_model(filename)

    def create_model(self):
        if self.mode == 'wide and deep':
            self.deep_component()
            self.wide_component()
            out_layer = concatenate([self.deep_component_outlayer,
                                     self.wide_component_outlayer])
            # inputs = [self.conti_input] + self.categ_inputs + [self.logistic_input]
            inputs = self.x_train

            output = Dense(1, activation='sigmoid')(out_layer)
            # Keras.models.model: https://keras.io/models/model/
            self.model = Model(inputs=inputs, outputs=output)
            return

        elif self.mode =='deep':
            # sequential model for categorical columns
            # https://www.tensorflow.org/tutorials/structured_data/feature_columns
            feature_layer = tf.keras.layers.DenseFeatures(
                feature_columns=self.deep_columns)
            self.model = tf.keras.Sequential([
                feature_layer,
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(1)
            ])


            # self.deep_component()
            # output = self.deep_component_outlayer
            # inputs = self.input_layer
            #
            # # Keras.models.model: https://keras.io/models/model/
            # self.model = Model(inputs=inputs, outputs=output)
            return
        else:
            print('wrong mode')
            return

    def train_model(self, epochs=1, optimizer='adam', batch_size=128):
        if not self.model:
            print('You have to create model first')
            return

        feature_all = CONF.get_feature_name()  # all features
        feature_used = CONF.get_feature_name('used')  # used features
        feature_unused = CONF.get_feature_name('unused')  # unused features
        feature_conf = CONF.read_feature_conf()  # feature conf dict
        feature_type = {}
        tensor_type = []
        for f in feature_all:
            if f in feature_conf:  # used features
                conf = feature_conf[f]
                if conf['type'] == 'category':
                    if conf['transform'] == 'identity':  # identity category column need int type
                        feature_type[f] = 'int32'
                        tensor_type.append(tf.int32)
                    else:
                        feature_type[f] = 'str'
                        tensor_type.append(tf.string)
                else:
                    feature_type[f] = 'float32'  # 0.0 for float32
                    tensor_type.append(tf.float32)
            else:  # unused features
                feature_type[f] = 'str'
                tensor_type.append(tf.string)
        feature_type['label'] = 'int32'
        tensor_type.append(tf.int32)
        table = pd.read_table("../data/train/train1", names=['label']+feature_all, dtype=feature_type)
        x, y = table[feature_all].copy(), table['label'].copy()
        dataset = tf.data.Dataset.from_tensor_slices((dict(x), y))
        # dataset = tf.data.Dataset.from_generator(table.iterrows(), output_types=feature_type)
        # model.fit: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        # self.model.fit(self.x_train_y_train, epochs=epochs, batch_size=batch_size,
        #                steps_per_epoch=1000)
        # self.model.fit(x.values, y.values, epochs=epochs, batch_size=batch_size,
        #                steps_per_epoch=1000)

        # WENQI
        # tutorial: https://www.tensorflow.org/tutorials/structured_data/feature_columns
        run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        run_metadata = tf.compat.v1.RunMetadata()
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'],
                           options=run_options, run_metadata=run_metadata)
        self.model.fit(dataset, epochs=epochs, steps_per_epoch=1000)

        from tensorflow.python.client import timeline
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('output/training_profiling.json', 'w') as f:
            f.write(ctf)

    def evaluate_model(self):
        if not self.model:
            print('You have to create model first')
            return

        loss, acc = self.model.evaluate(self.x_test, self.y_test)
        print(f'test_loss: {loss} - test_acc: {acc}')

    def predict_model(self):

        self.load_model()

        if not self.model:
            print('You have to create model first')
            return

        if self.mode == 'wide and deep':
            input_data = [self.x_test_conti] + \
                         [self.x_test_categ[:, i] for i in range(self.x_test_categ.shape[1])] + \
                         [self.x_test_categ_poly]
        elif self.mode == 'deep':
            input_data = [self.x_test_conti] + \
                         [self.x_test_categ[:, i] for i in range(self.x_test_categ.shape[1])]
        else:
            print('wrong mode')
            return

        print("Input data shape: {}".format(len(input_data)))

        # tensorboard --logdir=logs/scalars/
        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        run_metadata = tf.compat.v1.RunMetadata()
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'],
                           options=run_options, run_metadata=run_metadata)
        result = self.model.predict(input_data, batch_size=1, use_multiprocessing=True,
                                    callbacks=[tensorboard_callback])

        from tensorflow.python.client import timeline
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('output/inference_profiling.json', 'w') as f:
            f.write(ctf)

        print("result: {}".format(result))
        print("result shape：{}".format(result.shape))

    def save_model(self, filename='wide_and_deep.h5'):
        self.model.save(filename)


if __name__ == '__main__':

    FLAGS, unparsed = parser.parse_known_args()
    wide_deep_net = Wide_and_Deep("deep")

    train = True
    if train:
        wide_deep_net.create_model()
        wide_deep_net.train_model()
        # wide_deep_net.evaluate_model()
        wide_deep_net.save_model()
        plot_model(wide_deep_net.model, to_file='model.png', show_shapes=True, show_layer_names=False)

    # wide_deep_net.predict_model()
