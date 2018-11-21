# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import os
import tensorflow as tf
from configuration import config
from poros.poros_chars import tokenization
import csv
import collections

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None, "Input raw text file (or comma-separated list of files).")
flags.DEFINE_string("output_file", None, "TFRecord file (or comma-separated list of files).")
flags.DEFINE_string("vocab_file", None, "Vocab file")
flags.DEFINE_boolean("recreate_vocab", False, "if you want to create a vocab again?")
flags.DEFINE_boolean("is_lower", True, "make sure everything is lower case or not")
flags.DEFINE_integer("random_seed", 100, "Random seed")


def create_vocab(input_files, output_file, is_lower=True, skip_header=0):
    """

    :param input_files: a list of csv files
    :param skip_header: count of lines you would't read
    :return:
    """
    words = collections.OrderedDict()
    tokenizer = tokenization.BasicTokenizer(do_lower_case=is_lower)
    for file in input_files:
        with open(file) as f:
            rows = 0
            reader = csv.reader(f)
            for line in reader:
                rows += 1
                if rows <= skip_header:
                    tf.logging.info("header is {}".format(line))
                    continue
                text = tokenizer.tokenize(line[1])
                if rows <= 10:
                    tf.logging.info("line is {}, {}, {}".format(line[0], line[1], line[2]))
                    tf.logging.info("text is {}".format(text))
                for word in text:
                    words[word] = 0

    tf.logging.info("start write vocab, vocab size is {}".format(words.__len__()))
    with tf.gfile.GFile(output_file, mode="w") as writer:
        for word in words:
            writer.write(word)
            writer.write("\n")
        writer.write("<UNK>")


def generate_vocab(vocab_file):
    vocab = collections.OrderedDict()
    id = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            vocab[line] = id
            id += 1
    return vocab


def create_training_instance(input_files, vocab_file, is_lower=True, skip_header=0):
    """

    :param input_files: a list of csv files, such as ['a', 'b', 'c']
    :param vocab_file: vocab file
    :param is_lower: make sure all the words is in lower case or not
    :param skip_header: count of lines you would't read
    :return:
    """
    tokenizer = tokenization.BasicTokenizer(do_lower_case=is_lower)
    vocab = generate_vocab(vocab_file)
    instances = []
    for file in input_files:
        with open(file) as f:
            rows = 0
            reader = csv.reader(f)
            for line in reader:
                rows += 1
                if rows <= skip_header:
                    tf.logging.info("header is {}".format(line))
                    continue
                if rows <= 10:
                    tf.logging.info("line is {}, {}, {}".format(line[0], line[1], line[2]))
                text = tokenizer.tokenize(line[1])
                label = int(line[2])
                feature = collections.OrderedDict()
                feature["qid"] = line[0]
                feature["input_ids"] = tokenization.convert_tokens_to_ids(vocab, text)
                feature["label"] = label
                yield feature


def write_instances_to_example_files(instances, output_files):
    writer = []
    for file in output_files:
        f = tf.gfile.GFile(file, "w")
        writer.append(f)

    counts = 0
    for ele in instances:
        feature = collections.OrderedDict()
        feature["qid"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[ele["qid"].encode("utf-8")]))
        feature["input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=ele["input_ids"]))
        feature["label"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[ele["label"]]))
        if counts <= 10:
            tf.logging.info("qid is {}\n, input_ids is {}\n, label is {}".format(feature["qid"], feature["input_ids"], feature["label"]))

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer[counts % len(writer)].write(example.SerializeToString())
        counts += 1

    for w in writer:
        w.close()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    files = FLAGS.input_file.split(",")
    tf.logging.info("########## read files ###########")
    for file in files:
        tf.logging.info(file)

    vocab_file = FLAGS.vocab_file
    if not os.path.exists(vocab_file) or FLAGS.recreate_vocab:
        create_vocab(files, vocab_file, True, 1)

    instances = create_training_instance(files, vocab_file, True, 1)

    output_files = FLAGS.output_file.split(",")
    write_instances_to_example_files(instances, output_files)


if __name__ == "__main__":
    FLAGS.input_file = config.train_file
    FLAGS.output_file = config.output_file
    FLAGS.vocab_file = config.vocab_file
    FLAGS.recreate_vocab = False
    tf.app.run()
