# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import csv
from configuration import config
import tensorflow as tf
from poros.bert_model import create_pretraining_data


def generate_raw_data(skip_header=0):
    file_list = [config.train_file, config.test_file]
    writer = tf.gfile.GFile(config.bert_raw_data_file, "w")
    for file in file_list:
        with open(file, "r") as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                if i < skip_header:
                    continue
                writer.write(line[1])
                writer.write("\n")
    writer.close()


def generate_pretrain_data():
    create_pretraining_data.create_data(input_file=config.bert_raw_data_file,
                                        output_file=config.bert_intermediate_file,
                                        vocab_file=config.word_file, max_seq_length=config.max_sequence_length,
                                        max_predictions_per_seq=45)


def main(_):
    generate_raw_data(skip_header=1)
    generate_pretrain_data()


if __name__ == "__main__":
    s = input("即将重新生成预训练bert模型的tfrecord数据，如果数据已经存在，那么将会被覆盖，确认执行请输入yes\n")
    if s == "yes":
        tf.app.run()
