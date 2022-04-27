import codecs
import re
from collections import defaultdict
from random import choice, random

import numpy as np
from transformers import TF_DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST


def edit_distance(str1, str2):
    """Simple Levenshtein implementation for evalm."""
    table = np.zeros([len(str2) + 1, len(str1) + 1])
    for i in range(1, len(str2) + 1):
        table[i][0] = table[i - 1][0] + 1
    for j in range(1, len(str1) + 1):
        table[0][j] = table[0][j - 1] + 1
    for i in range(1, len(str2) + 1):
        for j in range(1, len(str1) + 1):
            if str1[j - 1] == str2[i - 1]:
                dg = 0
            else:
                dg = 1
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1, table[i - 1][j - 1] + dg)
    return int(table[len(str2)][len(str1)])


def get_char_acc(lstr1, lstr2):
    true_char = 0
    len_ref = len(lstr1)
    len_pred_here = len(lstr2)

    N = max(len_ref, len_pred_here)

    for y, y_hat in zip(lstr1, lstr1):
        if y == y_hat:
            true_char += 1

    acc_here = true_char / N
    return acc_here


def read_data(filename):
    with codecs.open(filename, "r", "utf-8") as inp:
        lines = inp.readlines()
    inputs = []
    outputs = []
    tags = []
    for l in lines:
        l = l.strip().split("\t")
        if l:
            inputs.append(list(l[0]))
            outputs.append(list(l[1]))
            tags.append(re.split("\W+", l[2]))

    return inputs, outputs, tags


def read_test_data(filename):
    with codecs.open(filename, "r", "utf-8") as inp:
        lines = inp.readlines()
    inputs = []
    tags = []
    for l in lines:
        l = l.strip().split("\t")
        if l:
            inputs.append(list(l[0]))
            tags.append(re.split("\W+", l[1]))
    return inputs, tags


def read_bpe_data(filename):
    with codecs.open(filename, "r", "utf-8") as inp:
        lines = inp.readlines()
    inputs = []
    outputs = []
    tags = []
    for l in lines:
        l = l.strip().split("\t")
        if l:
            inputs.append(l[0].strip().split())
            outputs.append(l[1].strip().split())
            tags.append(re.split("\W+", l[2]))
    return inputs, outputs, tags


def write_vocab(l, filename):
    with open(filename, "w") as outp:
        outp.write("\t".join(l))


def read_vocab(filename):
    with open(filename) as inp:
        lines = inp.readlines()
    return lines[0].strip().split("\t")


def argmax(arr, k):
    k = min(k, arr.size)
    # get k best indices
    indices = np.argpartition(arr, -k)[-k:]
    # sorted
    indices = indices[np.argsort(arr[indices])]
    # flip so first has largest value
    return np.flip(indices, 0)


def swap_io(items):
    return [(y, x) for (x, y) in items]


def get_chars(l):
    flat_list = [char for word in l for char in word]
    return list(set(flat_list))


def get_tags(l):
    flat_list = [tag for sublist in l for tag in sublist]
    return list(set(flat_list))


def _read_data(filename):
    with codecs.open(filename, "r", "utf-8") as inp:
        lines = inp.readlines()
    inputs = []
    outputs = []
    tags = []
    for l in lines:
        l = l.strip().split("\t")
        if l:
            inputs.append(l[0])
            outputs.append(l[1])
            tags.append(re.split("\W+", l[2]))

    return inputs, outputs, tags


def read_data_for_tokenizer(filename):
    with codecs.open(filename, "r", "utf-8") as inp:
        lines = inp.readlines()
    inputs = []
    outputs = []
    for l in lines:
        l = l.strip().split("\t")
        if l:
            inputs.append(l[0])
            outputs.append(l[1])

    return inputs, outputs


def make_data_for_tokenizer(train_path, dev_path, language):
    special_tokens = ["<pad>", "<s>", "</s>", "<copy>", "<unk>", "</tok>", "<mask>"]

    train_input, train_output = read_data_for_tokenizer(train_path)
    validate_input, validate_output = read_data_for_tokenizer(dev_path)
    raw_data = train_input + train_output + validate_input + validate_output

    with open(f"generated_data/{language}_data.txt", "w") as f:
        for item in raw_data:
            f.write("%s\n" % item)

    _, _, train_tags = _read_data(train_path)
    _, _, validate_tags = _read_data(dev_path)
    tags = get_tags(train_tags + validate_tags)
    for tag in tags:
        special_tokens.append("tag")

    return f"generated_data/{language}_data.txt", special_tokens


def read_data_bpe(filename):
    with codecs.open(filename, "r", "utf-8") as inp:
        lines = inp.readlines()
    inputs = []
    outputs = []
    tags = []
    for l in lines:
        l = l.strip().split("\t")
        if l:
            inputs.append(l[0])
            outputs.append(l[1])
            tags.append(" ".join(l[2].split(";")))

    return inputs, outputs, tags
    
