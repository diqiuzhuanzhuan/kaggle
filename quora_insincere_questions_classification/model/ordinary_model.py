# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
from poros.bert_model import run_classifier
import tensorflow as tf
from configuration import config
import csv
import random


def create_train_and_dev_file():

    header = None
    train = []
    dev = []
    with open(config.train_file, "r") as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                header = line
                continue
            if random.random() < 0.7:
                train.append(line)
            else:
                dev.append(line)
    with open(config.train_split_file, "w") as f:
        writer = csv.writer(f, delimiter=",", quotechar="\"")
        writer.writerow(header)
        for i in train:
            writer.writerow(i)

    with open(config.test_split_file, "w") as f:
        writer = csv.writer(f, delimiter=",", quotechar="\"")
        writer.writerow(header)
        for i in train:
            writer.writerow(i)


def main(_):
    tf.logging.set_verbosity()
    tf.logging.info(tf.logging.INFO)

    run_classifier.SimpleClassifierModel(
        bert_config_file=config.bert_config_file,
        vocab_file=config.word_file,
        output_dir="./output",
        max_seq_length=config.max_sequence_length,
        train_file=config.train_split_file,
        dev_file=config.dev_split_file,
        is_train=True
    )


if __name__ == "__main__":
    create_train_and_dev_file()
