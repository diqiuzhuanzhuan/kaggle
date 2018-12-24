# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import tensorflow as tf
from configuration import config


def model_fn_builder():

    def model_fn(features, labels, mode, params, config):
        tf.logging.info("our feature")
        tf.logging.info(features)
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    return model_fn


class SkyModel(object):

    def __init__(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def predict(self):
        pass