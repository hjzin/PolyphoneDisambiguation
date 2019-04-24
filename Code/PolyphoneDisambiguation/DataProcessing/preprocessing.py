#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@Time：2019/3/22
@Author: hhzjj
@Description：对数据进行预处理，包括tokenize、建立词典等
"""
import os
from torchtext.data import Field, TabularDataset
from torchtext.data import BucketIterator, Iterator
import DataProcessing.configure as config


def text_tokenize(x):
    """
    对text进行tokenize
    :param x: text
    :return: tokenize list
    """
    return [i for i in x]


def label_tokenize(y):
    """
    对label进行tokenize
    :param y: label集合
    :return: tokenize list
    """
    newlabel = list(eval(y))
    return newlabel


class BatchIterator:
    def __init__(self, train_path, valid_path, test_path, batch_size, format='csv'):
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.fomat = format
        self.TEXT = Field(sequential=True, use_vocab=True, tokenize=text_tokenize, batch_first=True)
        self.LABEL = Field(sequential=True, use_vocab=True, tokenize=label_tokenize, batch_first=True)

    def create_dataset(self):
        fields = [("text", self.TEXT), ("label", self.LABEL)]
        train, valid, test = TabularDataset.splits(
            path=os.getcwd(),
            train=self.train_path,
            validation=self.valid_path,
            test=self.test_path,
            format='csv',
            skip_header=True,
            fields=fields
        )
        self.TEXT.build_vocab(train)
        self.LABEL.build_vocab(train)
        return train, valid, test

    def get_iterator(self, train, valid, test):
        train_iter, val_iter, test_iter = BucketIterator.splits(
            (train, valid, test),
            batch_sizes=(self.batch_size, self.batch_size, self.batch_size),
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            shuffle=True
        )
        return train_iter, val_iter, test_iter


if __name__ == '__main__':
    batch_iter = BatchIterator(config.trn_file, config.val_file, config.tst_file,
                               batch_size=config.batch_size)
    train_data, valid_data, test_data = batch_iter.create_dataset()
    train_iter, valid_iter, test_iter = batch_iter.get_iterator(train=train_data, valid=valid_data, test=test_data)

