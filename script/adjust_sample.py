#!/usr/bin/env python
#encoding:utf8

import sys
import random

pos = []
neg = []

for line in sys.stdin:
    line = line.strip("\r\n")
    sps = line.split("\t")

    if len(sps) < 3:
        continue

    label = sps[1]

    if label == '0':
        neg.append(line)
    elif label == '1':
        pos.append(line)

print >>sys.stderr, "len(pos) = %d, len(neg) = %d" % (len(pos), len(neg))

if len(pos) > len(neg):
    tmp = pos
    pos = neg
    neg = tmp

sample_rate = 1.0 * len(pos) / len(neg)
for i in neg:
    if random.random() < sample_rate:
        pos.append(i)

random.shuffle(pos)

for i in pos:
    print i

sys.exit(0)

