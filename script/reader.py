#!/usr/bin/env python
#encoding:utf8

import os
import sys
import traceback
import random
import gzip

debug = False
class reader(object):
    def __init__(self, data_dir, sample_rate = 1):
        self.__data_dir = data_dir
        self.__sample_rate = sample_rate
    def __call__(self):
        for file_name in os.listdir(self.__data_dir):
            with open(os.path.join(self.__data_dir, file_name), "r") as fp:
                print >>sys.stderr, "loaded %s" % file_name
                for line in fp:
                    if random.random() > self.__sample_rate:
                        continue
                    line = line.strip("\r\n")
                    sps = line.split("\t")
                    if len(sps) != 3:
                        print >>sys.stderr, "bad line [%s]" % line
                        continue
                    try:
                        text = [int(i) for i in sps[0].split(",")]
                        label = int(sps[1])
                        weight = float(sps[2])
                    except:
                        traceback.print_exc(file = sys.stderr)
                        continue
                    yield [text, label, weight]

