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
