#!/usr/bin/env python
#encoding:utf8

import sys
import utils

char_dict = utils.load_dict("./data/char_dict")
unk_id = char_dict["<unk>"]
end_id = char_dict["</s>"]

for line in sys.stdin:
    line = line.strip("\r\n")
    sps = line.split("\t")

    text = sps[0].decode("utf8")
    ids = []
    for u in text:
        ids.append(char_dict.get(u.encode("utf8"), unk_id))
    ids.append(end_id)

    print "%s\t%s\t%s" % (",".join([str(i) for i in ids]), sps[1], sps[2])

