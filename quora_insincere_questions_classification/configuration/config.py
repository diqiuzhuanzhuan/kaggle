# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import os

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
embedding_file = os.path.join(data_path, "embeddings.zip")

glove_840b_300d = os.path.join(data_path, "glove.840B.300d", "glove.840B.300d.txt")
googlenews_vectors_gegative300 = os.path.join(data_path, "GoogleNews-vectors-negative300", "GoogleNews-vectors-negative300.bin")
paragram_300_sl999 = os.path.join(data_path, "paragram_300_sl999", "paragram_300_sl999.txt")
wiki_news_300d_1m = os.path.join(data_path, "wiki-news-300d-1M", "wiki-news-300d-1M.vec")

embedding_files = [glove_840b_300d, googlenews_vectors_gegative300, paragram_300_sl999, wiki_news_300d_1m]
for file in embedding_files:
    if not os.path.exists(file):
        print("文件不存在：{}".format(file))


"""
train, validation, test data
"""
train_file = os.path.join(data_path, "train.csv")
test_file = os.path.join(data_path, "test.csv")
for file in (train_file, test_file):
    if not os.path.exists(file):
        print("文件不存在：{}".format(file))

"""
vocab file
"""
vocab_file = os.path.join(data_path, "vocab.dict")
word_file = os.path.join(data_path, "word.dict")

"""
use word piece or not
"""
use_word = True

"""
output files used to store data preprocessed
"""
train_output_file = os.path.join(data_path, "train.feature_record.tf_record")
dev_output_file = os.path.join(data_path, "dev.feature_record.tf_record")
test_output_file = os.path.join(data_path, "test.feature_record.tf_record")

max_sequence_length = 300
vocab_size = 200344
word_size = 30522

random_seed = 10001

model_dir = os.path.join(data_path, "model")

"""
pretrain use bert
"""
bert_raw_data_file = os.path.join(data_path, "bert_raw_data.txt")
bert_intermediate_file = os.path.join(data_path, "bert_intermediate.tfrecord")
bert_config_file = os.path.join(data_path, "bert_config.json")
bert_model_path = os.path.join(model_dir, "bert_model")
bert_model_name = os.path.join(bert_model_path)
bert_train_batch_size = 8

"""
use text_cnn
"""
text_cnn_train_steps = 3000


"""
for ordinary model
"""
# firstly, we need a training file and dev file with csv format
train_split_file = os.path.join(data_path, "train_split_file.csv")
dev_split_file = os.path.join(data_path, "dev_split_file.csv")
ordinary_model_name = os.path.join(model_dir, "ordinary_model")
ordinary_test_result_file = os.path.join(ordinary_model_name, "ordinary_model_result.csv")
