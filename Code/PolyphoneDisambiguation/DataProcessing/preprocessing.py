#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@Time：2019/3/22
@Author: hhzjj
@Description：对数据进行预处理，包括tokenize、建立词典等
"""
import os
from torchtext.data import Field, Example, TabularDataset
from torchtext.data import BucketIterator
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
    def __init__(self, train_path, valid_path, batch_size, format='csv'):
        self.train_path = train_path
        self.valid_path = valid_path
        self.batch_size = batch_size
        self.fomat = format

    def create_dataset(self):
        TEXT = Field(sequential=True, use_vocab=True, tokenize=text_tokenize)
        LABEL = Field(sequential=True, use_vocab=True, tokenize=label_tokenize)

        fields = [("text", TEXT), ("label", LABEL)]
        train, valid = TabularDataset.splits(
            path=os.getcwd(),
            train=self.train_path,
            validation=self.valid_path,
            format='csv',
            skip_header=True,
            fields=fields
        )
        TEXT.build_vocab(train)
        LABEL.build_vocab(train)
        return train, valid

    def get_iterator(self, train, valid):
        train_iter, val_iter = BucketIterator.splits(
            (train, valid),
            batch_sizes=(self.batch_size, self.batch_size),
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            shuffle=True
        )
        return train_iter, val_iter


if __name__ == '__main__':
    bi = BatchIterator(train_path=config.trn_file, valid_path=config.val_file, batch_size=2)
    train, valid = bi.create_dataset()
    train_iter, valid_iter = bi.get_iterator(train=train, valid=valid)
    batch = next(iter(train_iter))
    print(len(train_iter))
    print('batch', batch)
    print('batch_text', batch.text)
    print('batch_label', batch.label)

