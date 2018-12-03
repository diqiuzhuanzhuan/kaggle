# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import pandas as pd
from configuration import config
import matplotlib.pyplot as plt


def get_data():
    train_data = pd.read_csv(config.train_file, index_col=0)
    test_data = pd.read_csv(config.test_file, index_col=0)
    print("train_data:")
    print(train_data.info())
    print(train_data.head(5))
    print("test_data:")
    print(test_data.info())
    print(test_data.head(5))
    return train_data, test_data


def main():
    train_data, test_data = get_data()
    target_stat = train_data["target"]
    plt.figure(0, figsize=(2, 6))
    plt.title("target")
    plt.hist(target_stat, 2, density=1, alpha=1)
    question_text_len_stat = [len(line) for line in train_data["question_text"]]
    plt.figure(1, figsize=(15, 6))
    plt.title("question length")
    plt.hist(question_text_len_stat, 50, density=1, facecolor='g', alpha=0.75)
    plt.show()


if __name__ == "__main__":
    main()

