from tokenize import Token

from data import encode
from src import Vocab


def generate_data(
    train_input,
    train_tags,
    validate_input,
    validate_tags,
    train_output,
    validate_output,
    vocab_x,
    vocab_y,
    tag_location,
):
    """Generate the data needed for the low_resource language task"""
    x_train = []

    for input, tag in zip(train_input, train_tags):
        if tag_location == "append":
            x_train.append(input + [Vocab.TOK] + tag)
        if tag_location == "prepend":
            x_train.append(tag + [Vocab.TOK] + input)

    x_validation = []
    for input, tag in zip(validate_input, validate_tags):
        if tag_location == "append":
            x_validation.append(input + [Vocab.TOK] + tag)
        if tag_location == "prepend":
            x_validation.append(tag + [Vocab.TOK] + input)

    study = []
    for x, y in zip(x_train, train_output):
        study.append((x, y))

    test = []
    for x, y in zip(x_validation, validate_output):
        test.append((x, y))

    max_x = len(max(x_train, key=len))
    max_y = len(max(train_output, key=len))
    train_items, test_items = encode(study, vocab_x, vocab_y), encode(test, vocab_x, vocab_y)

    val_items = test_items

    return train_items, test_items, val_items, study, test, max_x, max_y


def generate_data_for_tokenizer(
    train_input,
    train_tags,
    validate_input,
    validate_tags,
    train_output,
    validate_output,
    tokenizer,
):
    """Generate the data needed for the low_resource language task"""
    x_train = []
    for input, tag in zip(train_input, train_tags):
        x_train.append(input + " " + tag)

    x_validation = []
    for input, tag in zip(validate_input, validate_tags):
        x_validation.append(input + " " + tag)

    study = []
    for x, y in zip(x_train, train_output):
        study.append((x, y))

    test = []
    for x, y in zip(x_validation, validate_output):
        test.append((x, y))

    max_x = len(max(x_train, key=len))
    max_y = len(max(train_output, key=len))
    train_items, test_items = encode_bpe(study, test, tokenizer, max_x, max_y)
    val_items = test_items
    return train_items, test_items, val_items, study, test, max_x, max_y


def encode_bpe(study, test, tokenizer, max_x, max_y):
    train_items = []
    test_items = []

    for datum in study:
        text_encoding = tokenizer(
            datum[0],
            max_length=max_x,
            padding="max_length",
            add_special_tokens=True,
        )
        output_encoding = tokenizer(
            datum[1],
            max_length=max_y,
            padding="max_length",
            add_special_tokens=True,
        )
        train_items.append((text_encoding["input_ids"], output_encoding["input_ids"]))

    for datum in test:
        text_encoding = tokenizer(
            datum[0],
            max_length=max_x,
            padding="max_length",
            add_special_tokens=True,
        )
        output_encoding = tokenizer(
            datum[1],
            max_length=max_y,
            padding="max_length",
            add_special_tokens=True,
        )
        test_items.append((text_encoding["input_ids"], output_encoding["input_ids"]))

    return train_items, test_items
