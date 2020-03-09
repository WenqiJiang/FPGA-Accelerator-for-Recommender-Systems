from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import numpy as np
import pandas as pd

# fix ImportError: No mudule named lib.*
import sys
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from lib.read_conf import Config
CONF = Config()

parser = argparse.ArgumentParser(description='Train Wide and Deep Model.')
parser.add_argument(
    '--train_data', type=str, default=CONF.train["train_data"],
    help='Path to the train data.')
parser.add_argument(
    '--eval_data', type=str, default=CONF.train["eval_data"],
    help='Path to the validation data.')
parser.add_argument(
    '--pred_data', type=str, default=CONF.train["pred_data"],
    help='Path to the validation data.')
parser.add_argument(
    '--test_data', type=str, default=CONF.train["test_data"],
    help='Path to the test data.')
parser.add_argument(
    '--analyzed_data', type=str, default="../data/analyze/analyze1.csv",
    help='Path to the analyzed data.')
parser.add_argument(
    '--gen_data', type=bool, default=False,
    help='Whether to generate data from raw file')
FLAGS, unparsed = parser.parse_known_args()

def get_analyzed_columns(feature_conf_dic):
    """
    return a list of strings (col_name), only these columns will be analyzed.

    some of the features and all of the cross_features are stored in hash_buckets,
    which is unlikely to represent correlation between features.
    Thus, only those columns stored in meaningful format (vocab / id) are analyzed
    """
    # only keep columns that are not hashed, analyze correlation later
    keep_columns = []

    for feature, conf in feature_conf_dic.items():
        f_type, f_tran, f_param = conf["type"], conf["transform"], conf["parameter"]

        if f_type == 'category':

            if f_tran == 'hash_bucket':
                pass

            elif f_tran == 'vocab':
                keep_columns.append(feature)

            elif f_tran == 'identity':
                keep_columns.append(feature)

        # continuous features
        else:
            normalizaton, boundaries = f_param["normalization"], f_param["boundaries"]
            pass

    return keep_columns

def gen_analyzed_data():
    """
    Generate the data to be analyzed from the original pred data
    """
    # schemas
    SCHEMA = Config().read_schema()  # dict id -> col_name, e.g. SCHEMA[1]='clk'
    del SCHEMA[1]
    header_str = [v for k, v in SCHEMA.iteritems()]
    header_int = [k for k, v in SCHEMA.iteritems()]
    col2id = {v: k for k, v in SCHEMA.iteritems()}
    feature_conf_dic = CONF.read_feature_conf()
    cross_feature_list = CONF.read_cross_feature_conf()

    # load data
    df = pd.read_table(FLAGS.pred_data + "/pred1", header=header_int)

    # reformat the table, only analyzed columns are left
    keep_columns_str = get_analyzed_columns(feature_conf_dic)
    keep_columns_int = [col2id[v] for v in keep_columns_str]
    keep_columns_int.sort()
    df_keep_columns_int = [col - 2 for col in
                           keep_columns_int]  # dataframe starts from column 0; while our map start from 2
    analyzed_table = df.iloc[:, df_keep_columns_int]

    # save to csv
    analyzed_table.to_csv(FLAGS.analyzed_data, header=[SCHEMA[k] for k in keep_columns_int], index=False)
    print("Analyzed data generation finished.")

def gen_pred_csv():
    """
    Save the pred data as csv
    """
    # schemas
    SCHEMA = Config().read_schema()  # dict id -> col_name, e.g. SCHEMA[1]='clk'
    del SCHEMA[1]

    # load data
    df = pd.read_table(FLAGS.pred_data + "/pred1")

    # save to csv
    df.to_csv("../data/pred/pred1.csv", header=[v for k, v in SCHEMA.iteritems()], index=False)
    print("Csv generation finished.")

def gen_sample_csv():
    """
    Generate sample csv that contains both hashed and one-hot-encoded features
    """
    # schemas
    SCHEMA = Config().read_schema()  # dict id -> col_name, e.g. SCHEMA[1]='clk'
    del SCHEMA[1]

    # load data
    df = pd.read_csv("../data/pred/pred1.csv")

    # save to csv
    sample_col = ["request_id", "account_id", "adplan_id", "os", "client_type", "hour"]
    sample_table = df.loc[:, sample_col]
    sample_table.to_csv("../data/sample/sample.csv", header=sample_col, index=False)
    print("Csv generation finished.")


if __name__ == "__main__":

    # if FLAGS.gen_data:
    #     gen_analyzed_data()
    #
    # df = pd.read_csv(FLAGS.analyzed_data)
    # print("hi")
    # print("finished")

    gen_sample_csv()