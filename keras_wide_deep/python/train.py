import pandas as pd
import numpy as np
import argparse
import pandas as pd
from datetime import datetime


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
            return lambda x: tf.log(x)

    feature_conf_dic = CONF.read_feature_conf()
    cross_feature_list = CONF.read_cross_feature_conf()
    # tf.logging.info('Total used feature class: {}'.format(len(feature_conf_dic)))
    # tf.logging.info('Total used cross feature class: {}'.format(len(cross_feature_list)))

    wide_columns = []
    deep_columns = []
    wide_dim = 0
    deep_dim = 0
    feature_num = 0

    embedding_dimension_stat = []

    for feature, conf in feature_conf_dic.items():
        f_type, f_tran, f_param = conf["type"], conf["transform"], conf["parameter"]
        feature_num += 1
        if f_type == 'category':

            if f_tran == 'hash_bucket':
                hash_bucket_size = f_param
                embed_dim = embedding_dim(hash_bucket_size)
                embedding_dimension_stat.append((embed_dim, hash_bucket_size))
                col = categorical_column_with_hash_bucket(feature,
                    hash_bucket_size=hash_bucket_size,
                    dtype=tf.string)
                # wide_columns.append(col)
                # WENQI
                wide_columns.append(indicator_column(col))
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
                vocabulary_list = list(map(str, f_param))
                # col = categorical_column_with_vocabulary_list(feature,
                #     vocabulary_list=vocabulary_list,
                #     dtype=None,
                #     default_value=-1,
                #     num_oov_buckets=0)  # len(vocab)+num_oov_buckets
                # [16:33] 32 -> indicator + vocab -> error
                col = categorical_column_with_vocabulary_list(feature,
                    vocabulary_list=vocabulary_list,
                    dtype=None,
                    num_oov_buckets=1)  # len(vocab)+num_oov_buckets
                # WENQI
                # wide_columns.append(col)
                wide_columns.append(indicator_column(col))
                deep_columns.append(indicator_column(col))
                wide_dim += len(f_param)
                deep_dim += len(f_param)

            elif f_tran == 'identity':
                num_buckets = f_param
                col = categorical_column_with_identity(feature,
                    num_buckets=num_buckets,
                    default_value=0)  # Values outside range will result in default_value if specified, otherwise it will fail.
                # [33:36] -> indicator + id
                # WENQI
                # wide_columns.append(col)
                wide_columns.append(indicator_column(col))
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
                # WENQI
                wide_columns.append(bucketized_column(col, boundaries=boundaries))
                # wide_columns.append(indicator_column(bucketized_column(col, boundaries=boundaries)))
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
        # WENQI
        # wide_columns.append(col)
        wide_columns.append(indicator_column(col))
        wide_dim += hash_bucket_size
        if is_deep:
            deep_columns.append(embedding_column(col, dimension=embedding_dim(hash_bucket_size)))
            deep_dim += embedding_dim(hash_bucket_size)
            embedding_dimension_stat.append((embedding_dim(hash_bucket_size), hash_bucket_size))

    print("feature_num: {}\ncross_feature_num: {}\ntotal_feature_num: {}".format(
        feature_num, cross_feature_num, feature_num + cross_feature_num))

    print("embedding dimension:\nnum_embeddings: {}\taverage_dimension: {}\ntotal_param: {}".format(
        len(embedding_dimension_stat), np.average([dim for dim, size in embedding_dimension_stat]),
        np.sum([dim * size for dim, size in embedding_dimension_stat])))

    np.save("../cartesian_analysis/embedding_dimension", np.array(sorted(embedding_dimension_stat, key = lambda x: (x[0], x[1]))),
            allow_pickle=True, fix_imports=True)
    for dim, size in sorted(embedding_dimension_stat, key = lambda x: (x[0], x[1])):
        print("dim:{}\tsize:{}\t".format(dim, size))

    # add columns logging info
    tf.logging.info('Build total {} wide columns'.format(len(wide_columns)))
    for col in wide_columns:
        tf.logging.debug('Wide columns: {}'.format(col))
    tf.logging.info('Build total {} deep columns'.format(len(deep_columns)))
    for col in deep_columns:
        tf.logging.debug('Deep columns: {}'.format(col))
    tf.logging.info('Wide input dimension is: {}'.format(wide_dim))
    tf.logging.info('Deep input dimension is: {}'.format(deep_dim))

    # with tf.Session() as sess:
    #     sess.run(tf.tables_initializer())

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

    def get_dataset(self, mode="train", batch_size=32):
        """
        model = 'train' or 'eval' or 'pred'

        return: tf.data object
        """
        feature_all = CONF.get_feature_name()  # all features
        feature_conf = CONF.read_feature_conf()  # feature conf dict
        feature_type = {}
        for f in feature_all:
            if f in feature_conf:  # used features
                conf = feature_conf[f]
                if conf['type'] == 'category':
                    if conf['transform'] == 'identity':  # identity category column need int type
                        feature_type[f] = 'int32'
                    else:
                        feature_type[f] = 'str'
                else:
                    feature_type[f] = 'float32'  # 0.0 for float32
            else:  # unused features
                feature_type[f] = 'str'

        if mode == 'train' or mode == 'eval':
            feature_type['label'] = 'int32'
            if mode == 'train':
                table = pd.read_table("../data/train/train1", names=['label']+feature_all, dtype=feature_type)
            if mode == 'eval':
                table = pd.read_table("../data/eval/eval1", names=['label']+feature_all, dtype=feature_type)
            x, y = table[feature_all].copy(), table['label'].copy()
            if mode == 'evel':
                x, y = x.iloc[0: batch_size*5], y.iloc[0: batch_size*5]
            # drop unused
            for col in x.columns:
                if col not in feature_conf:
                    x.pop(col)
            # x_1, y_1 = x.iloc[0:5], y.iloc[0:5]
            dataset = tf.data.Dataset.from_tensor_slices((dict(x), y))
        elif mode == 'pred':
            table = pd.read_table("../data/pred/pred1", names=feature_all, dtype=feature_type)
            for col in table.columns:
                if col not in feature_conf:
                    table.pop(col)
            table = table.iloc[0: batch_size*1]
            dataset = tf.data.Dataset.from_tensor_slices(dict(table))
        else:
            raise("unrecognized mode!")

        dataset = dataset.batch(batch_size)

        return dataset

    def create_model(self):
        if self.mode == 'wide and deep':
            class MyModel(tf.keras.Model):

                def __init__(self, deep_col, wide_col):
                    super(MyModel, self).__init__()
                    self.deep_feature_layer = tf.keras.layers.DenseFeatures(feature_columns=deep_col)
                    self.deep_dense1 = tf.keras.layers.Dense(1024, activation='relu')
                    self.deep_dense2 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
                    self.deep_dense3 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
                    self.deep_dense4 = tf.keras.layers.Dense(1, activation=tf.nn.relu)

                    self.wide_feature_layer = tf.keras.layers.DenseFeatures(feature_columns=wide_col)
                    self.wide_dense = tf.keras.layers.Dense(1, activation=tf.nn.relu)

                    self.add_layer = tf.keras.layers.Add()
                    self.out_layer = tf.keras.layers.Softmax()

                def call(self, input_tensor):
                    d = self.deep_feature_layer(input_tensor)
                    d = self.deep_dense1(d)
                    d = self.deep_dense2(d)
                    d = self.deep_dense3(d)
                    d = self.deep_dense4(d)
                    w = self.wide_feature_layer(input_tensor)
                    w = self.wide_dense(w)
                    added = self.add_layer([w, d])
                    out = self.out_layer(added)

                    return out

            deep_col = self.deep_columns[:16] + self.deep_columns[33:]
            wide_col = self.wide_columns[:16] + self.wide_columns[36:]
            self.model = MyModel(deep_col, wide_col)

            return

        elif self.mode =='deep':
            # sequential model for categorical columns
            # https://www.tensorflow.org/tutorials/structured_../data/feature_columns
            class MyModel(tf.keras.Model):

                def __init__(self, deep_col):
                    super(MyModel, self).__init__()
                    self.feature_layer = tf.keras.layers.DenseFeatures(feature_columns=deep_col)
                    self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
                    self.norm1 = tf.keras.layers.BatchNormalization()
                    self.dense2 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
                    self.norm2 = tf.keras.layers.BatchNormalization()
                    self.dense3 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
                    self.norm3 = tf.keras.layers.BatchNormalization()
                    self.dense4 = tf.keras.layers.Dense(128, activation=tf.nn.softmax)

                def call(self, input_tensor):
                    y = self.feature_layer(input_tensor)
                    y = self.dense1(y)
                    y = self.norm1(y)
                    y = self.dense2(y)
                    y = self.norm2(y)
                    y = self.dense3(y)
                    y = self.norm3(y)
                    y = self.dense4(y)
                    # y = tf.random.categorical(
                    #     logits=y, num_samples=10, dtype=None, seed=None, name=None
                    # )

                    return y

            deep_col = self.deep_columns[:16] + self.deep_columns[33:]
            self.model = MyModel(deep_col)

            return
        elif self.mode == 'wide':
            wide_col = self.wide_columns[:16] + self.wide_columns[36:]
            # wide_col = self.wide_columns[36:39]
            # wide_col = self.wide_columns[39:]
            feature_layer = tf.keras.layers.DenseFeatures(
                feature_columns=wide_col)
            self.model = tf.keras.Sequential([
                feature_layer,
                Dense(1, activation='softmax')
            ])
            return
        else:
            print('wrong mode')
            return

    def train_model(self, epochs=1, optimizer='adam', batch_size=128):
        # if not self.model:
        #     print('You have to create model first')
        #     return

        dataset = self.get_dataset(mode="train", batch_size=batch_size)
        # model.fit: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        # WENQI
        # tutorial: https://www.tensorflow.org/tutorials/structured_../data/feature_columns
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'],
                           options=run_options, run_metadata=run_metadata)
        plot_model(self.model, to_file='model.png', show_shapes=True,
            show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)

        dataset_iter = self.x_train_y_train.make_initializable_iterator()
        next_element = dataset_iter.get_next()

        sess = tf.Session()
        sess.run([tf.local_variables_initializer(), tf.tables_initializer()])
        sess.run(dataset_iter.initializer)
        data = sess.run(next_element)
        tf.keras.backend.set_session(sess)
        tf.keras.backend.get_session().run(tf.tables_initializer(name='init_all_tables'))

        # with tf.Session() as sess:
        #     sess.run([tf.local_variables_initializer(), tf.tables_initializer()])
        #     sess.run(dataset_iter.initializer)
        #     data = sess.run(next_element)
        #     tf.keras.backend.set_session(sess)
        #     tf.keras.backend.get_session().run(tf.tables_initializer(name='init_all_tables'))
        self.model.fit(dataset, epochs=epochs, steps_per_epoch=1)
            # self.model.fit(self.x_train_y_train, epochs=epochs, steps_per_epoch=1000)

        from tensorflow.python.client import timeline
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('../output/training_profiling.json', 'w') as f:
            f.write(ctf)

    def evaluate_model(self):
        if not self.model:
            print('You have to create model first')
            return

        dataset = self.get_dataset(mode="eval", batch_size=32)
        loss, acc = self.model.evaluate(dataset)
        print(f'test_loss: {loss} - test_acc: {acc}')

    def predict_model(self):

        if not self.model:
            self.load_model()

        # NOTE! CHANGE HERE TO ADJUST INFERENCE BATCH SIZE
        batch_size = 1024
        input_data = self.get_dataset(mode="pred", batch_size=batch_size)

        # print("Input data shape: {}".format(len(input_data)))

        # tensorboard --logdir=logs/scalars/
        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'],
                           options=run_options, run_metadata=run_metadata)
        result = self.model.predict(input_data, use_multiprocessing=True,
                                    callbacks=[tensorboard_callback])

        from tensorflow.python.client import timeline
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()

        if self.mode == "wide and deep":
            filename = '../output/{}_inference_profiling_batch_{}.json'.format("wide_deep", batch_size)
        elif self.mode == "wide":
            filename = '../output/{}_inference_profiling_batch_{}.json'.format("wide", batch_size)
        elif self.mode == "deep":
            filename = '../output/{}_inference_profiling_batch_{}.json'.format("deep", batch_size)
        with open(filename, 'w') as f:
            f.write(ctf)

        print("result: {}".format(result))
        print("result shape：{}".format(result.shape))

    def save_model(self, filename='../model/wide_and_deep'):
        # self.model.save(filename)
        self.model.save_weights(filepath=filename, overwrite=True, save_format='h5')
        # self.model.save_weights(filepath=filename, overwrite=True, save_format='tf')

    def load_model(self, filename='../model/wide_and_deep'):
        self.create_model()
        self.model.load_weights(filepath=filename)

if __name__ == '__main__':

    # mode = "wide and deep"
    mode = "deep"
    FLAGS, unparsed = parser.parse_known_args()
    wide_deep_net = Wide_and_Deep(mode)

    train = True
    if train:
        wide_deep_net.create_model()
        wide_deep_net.train_model()
        wide_deep_net.save_model()
        if mode == "deep":
            wide_deep_net.evaluate_model()
        plot_model(wide_deep_net.model, to_file='model.png', show_shapes=True, show_layer_names=False)

    wide_deep_net.predict_model()
