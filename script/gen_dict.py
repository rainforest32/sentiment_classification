#!/usr/bin/env python
#encoding:utf8

import sys

char_set = set()
for line in sys.stdin:
    line = line.strip("\r\n")
    sps = line.split("\t")

    text = sps[0].decode("utf8")
    for u in text:
        char_set.add(u)
print >>sys.stderr, "len of char_set is %d" % len(char_set)

i = 0
print "%s\t%d" % ("<unk>", i)
i += 1
print "%s\t%d" % ("</s>", i)
i += 1

for u in char_set:
    print "%s\t%d" % (u.encode("utf8"), i)
    i += 1
sys.exit(0)

