# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
from poros.text_cnn import text_cnn
from poros.poros_train import optimization
import tensorflow as tf
from configuration import config


class EarthConfig(object):
    max_sequence_length = config.max_sequence_length
    num_classes = 2
    vocab_size = config.vocab_size
    embedding_size = 256
    filter_sizes = [2, 3, 4, 5]
    num_filters = 3
    lr = 5e-5
    train_batch_size = 64
    dev_batch_size = 128
    predict_batch_size = 128


def model_fn_builder(earth_config=EarthConfig()):

    def model_fn(features, labels, mode, params, config):
        tf.logging.info("our feature:")
        tf.logging.info(features)
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        qid = features["qid"]
        input_x = features["input_ids"]
        tf.logging.info(input_x)
        input_y = tf.one_hot(features["label"], depth=2)

        if tf.estimator.ModeKeys.TRAIN == mode:
            is_training = True
        else:
            is_training = False

        model = text_cnn.TextCNN(
            input_x,
            input_y,
            is_training,
            sequence_length=earth_config.max_sequence_length,
            num_classes=earth_config.num_classes,
            vocab_size=earth_config.vocab_size,
            embedding_size=earth_config.embedding_size,
            filter_sizes=earth_config.filter_sizes,
            num_filters=earth_config.num_filters,
        )
        spec = {}
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(model.loss, init_lr=earth_config.lr, num_train_steps=10000, num_warmup_steps=None, use_tpu=None)
            spec = tf.estimator.EstimatorSpec(loss=model.loss, train_op=train_op, mode=mode)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics_op = {
                "accuracy": tf.metrics.accuracy(labels=features["label"], predictions=model.predictions),
                "recall": tf.metrics.recall(labels=features["label"], predictions=model.predictions),
                "f1_score": tf.contrib.metrics.f1_score(labels=features["label"], predictions=model.predictions),
                "average_loss": tf.metrics.mean(model.loss)
            }
            spec = tf.estimator.EstimatorSpec(loss=model.loss, eval_metric_ops=eval_metrics_op, mode=mode)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            spec = tf.estimator.EstimatorSpec(predictions=model.predictions, mode=mode)
        else:
            spec = tf.estimator.EstimatorSpec(predictions=model.predictions)

        return spec

    return model_fn


def train_input_fn_builder(batch_size):

    def input_fn():
        name_to_feature = {
            "qid": tf.FixedLenFeature([], dtype=tf.string),
            "input_ids": tf.FixedLenFeature([config.max_sequence_length], dtype=tf.int64),
            "label": tf.FixedLenFeature([], dtype=tf.int64)
        }
        filenames = config.train_output_file.split(',')
        tf.logging.info(filenames)

        def _decode(record):
            tf.logging.info("****************")
            tf.logging.info(record)
            return tf.parse_single_example(record, name_to_feature)

        dataset = tf.data.TFRecordDataset(filenames=config.train_output_file.split(','))
        dataset = dataset.map(lambda record: _decode(record)).batch(batch_size=batch_size)
        return dataset

    return input_fn


def dev_input_fn_builder(batch_size):

    def input_fn():
        name_to_feature = {
            "qid": tf.FixedLenFeature([], dtype=tf.string),
            "input_ids": tf.FixedLenFeature([config.max_sequence_length], dtype=tf.int64),
            "label": tf.FixedLenFeature([], dtype=tf.int64)
        }
        filenames = config.dev_output_file.split(',')
        tf.logging.info(filenames)

        def _decode(record):
            tf.logging.info("****************")
            tf.logging.info(record)
            return tf.parse_single_example(record, name_to_feature)

        dataset = tf.data.TFRecordDataset(filenames=config.train_output_file.split(','))
        dataset = dataset.map(lambda record: _decode(record)).batch(batch_size=batch_size)
        return dataset

    return input_fn


def predict_input_fn_builder(batch_size):

    def input_fn():
        name_to_feature = {
            "qid": tf.FixedLenFeature([], dtype=tf.string),
            "input_ids": tf.FixedLenFeature([config.max_sequence_length],dtype=tf.int64),
            "label": tf.FixedLenFeature([], dtype=tf.int64)
        }

        dataset = tf.data.TFRecordDataset(filenames=config.test_output_file.split(','))
        dataset = dataset.map(
            lambda record: tf.parse_single_example(record, name_to_feature)
        ).batch(batch_size=batch_size)

        return dataset

    return input_fn


class EarthModel(object):

    def __init__(self):
        self.earth_config = EarthConfig()
        run_config = tf.estimator.RunConfig()
        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn_builder(earth_config=self.earth_config),
            model_dir=config.model_dir,
            config=run_config,
            params={}
        )

    def train(self, steps=config.text_cnn_train_steps):
        training = self.estimator.train(input_fn=train_input_fn_builder(self.earth_config.train_batch_size),
                                        steps=steps)
        return training

    def eval(self):
        evaluation = self.estimator.evaluate(input_fn=dev_input_fn_builder(self.earth_config.dev_batch_size))
        return evaluation

    def predict(self):
        """

        :return:  like this, [1, 0, 1, 0]
        """
        prediction = self.estimator.predict(input_fn=predict_input_fn_builder(self.earth_config.predict_batch_size))
        return prediction


def eval():
    earth_model = EarthModel()
    evaluation = earth_model.eval()
    for i, j in evaluation.items():
        tf.logging.info("{}, {}".format(i, j))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    eval()
