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

    if tf.gfile.Exists(config.train_split_file) and tf.gfile.Exists(config.dev_split_file):
        tf.logging.info("{} and {} are existed, don't recreate!")
        tf.logging.info("if you want to recreate, please delete them")
        return
    train = []
    dev = []
    with open(config.train_file, "r") as f:
        reader = csv.reader(f, delimiter=",", quotechar="\"")
        for i, line in enumerate(reader):
            if i == 0:
                """
                we don't need header
                """
                continue
            if random.random() < 0.7:
                train.append(line)
            else:
                dev.append(line)
    with open(config.train_split_file, "w") as f:
        writer = csv.writer(f, delimiter=",", quotechar="\"")
        for i in train:
            writer.writerow(i)

    with open(config.dev_split_file, "w") as f:
        writer = csv.writer(f, delimiter=",", quotechar="\"")
        for i in train:
            writer.writerow(i)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("start...")

    model = run_classifier.SimpleClassifierModel(
        bert_config_file=config.bert_config_file,
        vocab_file=config.word_file,
        output_dir="ordinary_model",
        max_seq_length=config.max_sequence_length,
        train_file=config.train_split_file,
        dev_file=config.dev_split_file,
        is_train=True,
        label_list=["0", "1"],
        init_checkpoint=config.bert_model_name
    )
    model.train()


if __name__ == "__main__":
    create_train_and_dev_file()
    main(None)