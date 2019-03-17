#!/usr/bin/env python
#encoding:utf8

import sys

test_num = 10000
dev_num = 10000

i = 0

test_fp = open("./data/test_data.all", "w")
dev_fp = open("./data/dev_data.all", "w")
train_fp = open("./data/train_data.all", "w")

for line in sys.stdin:
    line = line.strip("\r\n")
    fp = train_fp
    if i < test_num:
        fp = test_fp
    elif i < test_num + dev_num:
        fp = dev_fp
    print >>fp, line
    i += 1
sys.exit(0)

