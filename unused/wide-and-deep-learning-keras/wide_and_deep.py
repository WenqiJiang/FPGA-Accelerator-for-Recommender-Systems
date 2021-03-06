import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder

from datetime import datetime
import tensorflow as tf
from tensorflow import keras
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, SpatialDropout1D, Activation, concatenate
from keras.optimizers import Adam, SGD
from keras.layers.advanced_activations import ReLU, PReLU, LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model
from tensorflow.keras.utils import plot_model

from lib.read_conf import Config

def preprocessing():
    train_data = pd.read_csv('./adult.data', names=COLUMNS)
    train_data.dropna(how='any', axis=0)
    test_data = pd.read_csv('./adult.test', skiprows=1, names=COLUMNS)
    test_data.dropna(how='any', axis=0)
    all_data = pd.concat([train_data, test_data])
    # ラベルを数値化する
    all_data[LABEL_COLUMN] = all_data['income_bracket'].apply(lambda x: ">50K" in x).astype(int)
    all_data.pop('income_bracket')
    y = all_data[LABEL_COLUMN].values
    all_data.pop(LABEL_COLUMN)
    for c in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        all_data[c] = le.fit_transform(all_data[c])
    train_size = len(train_data)
    x_train = all_data.iloc[:train_size]
    y_train = y[:train_size]
    x_test = all_data.iloc[train_size:]
    y_test = y[train_size:]
    x_train_categ = np.array(x_train[CATEGORICAL_COLUMNS]) # カテゴリーデータ
    x_test_categ = np.array(x_test[CATEGORICAL_COLUMNS])
    x_train_conti = np.array(x_train[CONTINUOUS_COLUMNS], dtype='float64') # 連続的データ
    x_test_conti = np.array(x_test[CONTINUOUS_COLUMNS], dtype='float64')
    scaler = StandardScaler()
    x_train_conti = scaler.fit_transform(x_train_conti) # 連続データの訓練セットの平均とstdで標準化
    x_test_conti = scaler.transform(x_test_conti)
    return [x_train, y_train, x_test, y_test, x_train_categ, x_test_categ, x_train_conti, x_test_conti, all_data]


class Wide_and_Deep:
    def __init__(self):

        self._conf = Config()
        self._train_conf = self._conf.train
        self._cnn_conf = self._conf.model

        x_train, y_train, x_test, y_test, x_train_categ, x_test_categ, x_train_conti, x_test_conti, all_data \
            = preprocessing()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_train_categ = x_train_categ # 訓練セットの中のカテゴリーデータ
        self.x_test_categ = x_test_categ # テストセットの中のカテゴリーデータ
        self.x_train_conti = x_train_conti # 訓練セットの中の連続的データ
        self.x_test_conti = x_test_conti # テストセットの中の連続的データ
        self.all_data = all_data
        self.poly = PolynomialFeatures(degree=2, interaction_only=True)
        # カテゴリーデータをcross product化
        self.x_train_categ_poly = self.poly.fit_transform(x_train_categ)
        self.x_test_categ_poly = self.poly.transform(x_test_categ)
        self.categ_inputs = None
        self.conti_input = None
        self.deep_component_outlayer = None
        self.logistic_input = None
        self.model = None

    def deep_component(self):
        categ_inputs = []
        categ_embeds = []
        # カテゴリーデータの特徴ごとにInput層とEmbedding層を作成
        for i in range(len(CATEGORICAL_COLUMNS)):
            input_i = Input(shape=(1,), dtype='int32')
            dim = len(np.unique(self.all_data[CATEGORICAL_COLUMNS[i]]))
            embed_dim = int(np.ceil(dim ** 0.25)) # 入力カテゴリー数の4乗根をEmbedding次元数にする
            embed_i = Embedding(dim, embed_dim, input_length=1)(input_i)
            flatten_i = Flatten()(embed_i)
            categ_inputs.append(input_i)
            categ_embeds.append(flatten_i)
        # 連続的データは全結合層で一括入力
        conti_input = Input(shape=(len(CONTINUOUS_COLUMNS),))
        # conti_dense = Dense(256, use_bias=False)(conti_input)
        # 全結合層と各Embeddingの出力をくっつける
        concat_embeds = concatenate([conti_input]+categ_embeds)
        concat_embeds = Activation('relu')(concat_embeds)
        bn_concat = BatchNormalization()(concat_embeds)
        # 更に全結合層を3層重ねる
        fc1 = Dense(1024, use_bias=False)(bn_concat)
        ac1 = ReLU()(fc1)
        bn1 = BatchNormalization()(ac1)
        fc2 = Dense(512, use_bias=False)(bn1)
        ac2 = ReLU()(fc2)
        bn2 = BatchNormalization()(ac2)
        fc3 = Dense(256)(bn2)
        ac3 = ReLU()(fc3)
        # WENQI
        deep_out = Dense(1)(ac3)

        self.categ_inputs = categ_inputs
        self.conti_input = conti_input
        # WENQI
        # self.deep_component_outlayer = ac3
        self.deep_component_outlayer = deep_out

    def wide_component(self):
        # カテゴリーデータだけ線形モデルに入れる
        dim = self.x_train_categ_poly.shape[1]
        self.logistic_input = Input(shape=(dim,))
        self.wide_component_outlayer = Dense(1)(self.logistic_input)

    def load_model(self, filename='wide_and_deep.h5'):
        self.model = load_model(filename)

    def create_model(self):
        self.deep_component()
        self.wide_component()
        if self.mode == 'wide and deep':
            out_layer = concatenate([self.deep_component_outlayer,
                                     self.wide_component_outlayer])
            inputs = [self.conti_input] + self.categ_inputs + [self.logistic_input]
        elif self.mode =='deep':
            out_layer = self.deep_component_outlayer
            inputs = [self.conti_input] + self.categ_inputs
        else:
            print('wrong mode')
            return

        output = Dense(1, activation='sigmoid')(out_layer)
        self.model = Model(inputs=inputs, outputs=output)

    def train_model(self, epochs=0, optimizer='adam', batch_size=128):
        if not self.model:
            print('You have to create model first')
            return

        if self.mode == 'wide and deep':
            input_data = [self.x_train_conti] +\
                         [self.x_train_categ[:, i] for i in range(self.x_train_categ.shape[1])] +\
                         [self.x_train_categ_poly]
        elif self.mode == 'deep':
            input_data = [self.x_train_conti] +\
                         [self.x_train_categ[:, i] for i in range(self.x_train_categ.shape[1])]
        else:
            print('wrong mode')
            return

        # WENQI
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'],
                           options=run_options, run_metadata=run_metadata)
        self.model.fit(input_data, self.y_train, epochs=epochs, batch_size=batch_size)

        from tensorflow.python.client import timeline
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('output/training_profiling.json', 'w') as f:
            f.write(ctf)

    def evaluate_model(self):
        if not self.model:
            print('You have to create model first')
            return

        if self.mode == 'wide and deep':
            input_data = [self.x_test_conti] +\
                         [self.x_test_categ[:, i] for i in range(self.x_test_categ.shape[1])] +\
                         [self.x_test_categ_poly]
        elif self.mode == 'deep':
            input_data = [self.x_test_conti] +\
                         [self.x_test_categ[:, i] for i in range(self.x_test_categ.shape[1])]
        else:
            print('wrong mode')
            return

        loss, acc = self.model.evaluate(input_data, self.y_test)
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

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'],
                           options=run_options, run_metadata=run_metadata)
        result = self.model.predict(input_data, batch_size=128, use_multiprocessing=True,
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
    wide_deep_net = Wide_and_Deep()

    train = False
    if train:
        wide_deep_net.create_model()
        wide_deep_net.train_model()
        wide_deep_net.evaluate_model()
        wide_deep_net.save_model()
        plot_model(wide_deep_net.model, to_file='model.png', show_shapes=True, show_layer_names=False)

    wide_deep_net.predict_model()
