############ This program is not successful ##############

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
from train import Wide_and_Deep

class Wide_and_Deep_Intermediate_Output(Wide_and_Deep):

    def __init__(self, mode="deep"):
        super().__init__(mode)

    def predict_intermediate(self, layer_name="dense_1"):

        if not self.model:
            self.load_model()

        input_data = self.get_dataset(mode="pred", batch_size=128)
        # print("Input data shape: {}".format(len(input_data)))

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        intermediate_layer_model = tf.keras.Model(inputs=self.model.input,
                                         outputs=self.model.get_layer(layer_name).output)
        result = intermediate_output = intermediate_layer_model.predict(input_data)

        print("result: {}".format(result))
        print("result shape：{}".format(result.shape))

if __name__ == '__main__':

    # mode = "wide and deep"
    mode = "deep"
    # wide_deep_net = Wide_and_Deep_Intermediate_Output(mode)
    wide_deep_net = Wide_and_Deep(mode)
    wide_deep_net.load_model()
    get_3rd_layer_output = tf.keras.backend.function([wide_deep_net.model.layers[0].input], [wide_deep_net.model.layers[3].output])
    layer_output = get_3rd_layer_output([x])[0]
    # wide_deep_net.predict_model()
    # wide_deep_net.predict_intermediate()
