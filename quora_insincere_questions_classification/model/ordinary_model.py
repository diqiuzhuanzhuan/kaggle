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

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_bool("eval", True, "eval for evaluation file")
tf.flags.DEFINE_bool("train", True, "train for training file ")
tf.flags.DEFINE_bool("test", True, "predict for test file")
tf.flags.DEFINE_bool("recreate_data", False, "generate train and dev data no matter the data is existed or not")


def create_train_and_dev_file(force):

    if tf.gfile.Exists(config.train_split_file) and tf.gfile.Exists(config.dev_split_file) and not force:
        tf.logging.info("{} and {} are existed, don't recreate!".format(config.train_split_file, config.dev_split_file))
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


def read_test_file():
    test_data = []
    qid = []
    with open(config.test_file, "r") as f:
        reader = csv.reader(f, delimiter=",", quotechar="\"")
        for i, line in enumerate(reader):
            if i == 0:
                continue

            test_data.append([str(i), line[1]])
            qid.append(line[0])

    return test_data, qid


def write_result(test_result, qid):
    with open(config.ordinary_test_result_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "prediction"])
        for i, j in zip(qid, test_result):
            writer.writerow([i, j])


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("start...")

    create_train_and_dev_file(force=FLAGS.recreate_data)

    init_checkpoint = config.bert_model_name
    model = run_classifier.SimpleClassifierModel(
        train_batch_size=config.bert_train_batch_size,
        bert_config_file=config.bert_config_file,
        vocab_file=config.word_file,
        output_dir=config.ordinary_model_name,
        max_seq_length=config.max_sequence_length,
        train_file=config.train_split_file,
        dev_file=config.dev_split_file,
        is_train=True,
        label_list=["0", "1"],
        init_checkpoint=init_checkpoint
    )
    if FLAGS.train:
        model.train()
    if FLAGS.eval:
        model.eval()
    if FLAGS.test:
        test_data, qid = read_test_file()
        res = model.predict(test_data)
        write_result(res, qid)


if __name__ == "__main__":
    tf.app.run()
