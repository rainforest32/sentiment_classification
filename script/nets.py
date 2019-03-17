#!/usr/bin/env python

import sys
import math
import gzip
import time
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.param_attr as attr
import lstm_attention

def bilstm_attention_classify_net(dict_size = 12232,
        emb_size = 1024,
        lstm_size = 1024,
        class_size = 2,
        drop_rate = 0.5,
        is_test = False,
        is_py_reader = False):
    if is_py_reader:
        reader = fluid.layers.py_reader(capacity = 10240,
                shapes = [[-1, 1], [-1, 1], [-1, 1]],
                lod_levels = [1, 0, 0],
                dtypes = ['int64', 'int64', 'float32'],
                name = "test_reader" if is_test else "train_reader",
                use_double_buffer = True)
        text, label, weight = fluid.layers.read_file(reader)
    else:
        text = fluid.layers.data(name='text', 
                shape=[1], 
                dtype='int64', 
                lod_level = 1)
        label = fluid.layers.data(name='label', 
                shape=[1], 
                dtype='int64',
                lod_level = 0)
        weight = fluid.layers.data(name='weight', 
                shape=[1], 
                dtype='float32', 
                lod_level = 0)

    text_emb = fluid.layers.embedding(input = text,
            size = [dict_size, emb_size],
            is_sparse = True)
    lstm_attention_model = lstm_attention.LSTMAttentionModel(
            lstm_size = lstm_size, 
            drop_rate = drop_rate)
    lstm = fluid.layers.relu(lstm_attention_model.forward(text_emb, not is_test))
    predict = fluid.layers.fc(input = lstm, 
            size = class_size, 
            act = 'softmax')
    cost = fluid.layers.elementwise_mul(
            x = fluid.layers.cross_entropy(input = predict, label = label),
            y = weight,
            axis = 0)
    avg_cost = fluid.layers.mean(x = cost)
    acc = fluid.layers.accuracy(input=predict, label=label)

    if is_py_reader:
        return [avg_cost, acc, predict, reader]
    else:
        return [avg_cost, acc, predict]

