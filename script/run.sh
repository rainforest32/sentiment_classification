#!/usr/bin/env sh

. ~/.wangyou_rc
. $common/script/common.sh

awk -F"\t" 'BEGIN {
    OFS = "\t";
} {
    text = $2;
    if (length(text) < 4) {
        next;
    }

    if ($3 == 3) {
        next;
    } else if ($3 == 1) {
        label = 0;
        weight = 1.0;
    } else if ($3 == 2) {
        label = 0;
        weight = 0.8;
    } else if ($3 == 4) {
        label = 1;
        weight = 0.8;
    } else if ($3 == 5) {
        label = 1;
        weight = 1.0;
    } else {
        print "bad line: "$0 >"/dev/stderr";
    }
    print text, label, weight;
}' data/douban_comment.txt | sort -u -S2g -T. >data/all.tsv
check_ret $? "gen all.tsv fail"

# 样本均衡
python script/adjust_sample.py <data/all.tsv >data/all_adjust.tsv
check_ret $? "adjust sample fail"

python script/gen_dict.py <data/all_adjust.tsv >data/char_dict
check_ret $? "get char_dict fail"

python script/gen_train_data.py <data/all_adjust.tsv >data/ids_data
check_ret $? "get ids_data fail"

python script/split_train_data.py <data/ids_data
check_ret $? "get split data fail"

mkdir -p data/train_data
mkdir -p data/test_data

rm -f data/train_data/*
rm -f data/test_data/*

cp data/train_data.all data/train_data/train_data

cp data/test_data.all data/test_data/test_data

CUDA_VISIBLE_DEVICES=0 FLAGS_fraction_of_gpu_memory_to_use=0.5 python script/train.py >train.log 2>&1

