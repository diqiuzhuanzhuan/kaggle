# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import csv
from configuration import config
import tensorflow as tf
from poros.bert_model import create_pretraining_data, pretrain
import random


def generate_raw_data(skip_header=0):
    rng = random.Random(config.random_seed)
    file_list = [config.train_file, config.test_file]
    writer = tf.gfile.GFile(config.bert_raw_data_file, "w")
    for file in file_list:
        with open(file, "r") as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                if i < skip_header:
                    continue
                line = line[1]
                line = line.split(" ")
                if len(line) > 30:
                    pos = rng.randint(1, len(line))
                    writer.write(" ".join(line[0:pos]))
                    writer.write("\n")
                    writer.write(" ".join(line[pos:]))
                    writer.write("\n")
                else:
                    writer.write(" ".join(line))
                    writer.write("\n")
                writer.write("\n")
    writer.close()


def generate_pretrain_data():
    create_pretraining_data.create_data(input_file=config.bert_raw_data_file,
                                        output_file=config.bert_intermediate_file,
                                        vocab_file=config.word_file, max_seq_length=config.max_sequence_length,
                                        max_predictions_per_seq=45)


def train():
    pretrain.run(input_file=config.bert_intermediate_file,
                 bert_config_file=config.bert_config_file,
                 output_dir=config.bert_model_path,
                 max_predictions_per_seq=45,
                 max_seq_length=config.max_sequence_length,
                 do_train=True,
                 do_eval=True,
                 train_batch_size=config.bert_train_batch_size,
                 num_train_steps=100,
                 alpha=0.8,
                 beta=0.2,
                )


def data_prepare():
    generate_raw_data(skip_header=1)
    generate_pretrain_data()


if __name__ == "__main__":
    s = input("即将重新生成预训练bert模型的tfrecord数据，如果数据已经存在，那么将会被覆盖，确认执行请输入yes, 跳过请输入其他.\n")
    if s == "yes":
        data_prepare()
    s = input("即将开始预训练，确认请输入yes\n")
    if s == "yes":
        train()
