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
    test_input,
    test_tags,
    test_output,
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

    x_test = []
    for input, tag in zip(test_input, test_tags):
        if tag_location == "append":
            x_test.append(input + [Vocab.TOK] + tag)
        if tag_location == "prepend":
            x_test.append(tag + [Vocab.TOK] + input)

    study = []
    for x, y in zip(x_train, train_output):
        study.append((x, y))

    validate = []
    for x, y in zip(x_validation, validate_output):
        validate.append((x, y))

    test = []
    for x, y in zip(x_test, test_output):
        test.append((x, y))

    max_x = len(max(x_train, key=len))
    max_y = len(max(train_output, key=len))
    train_items, validate_items = encode(study, vocab_x, vocab_y), encode(
        validate, vocab_x, vocab_y
    )

    test_items = encode(test, vocab_x, vocab_y)

    return train_items, validate_items, test_items, study, test, validate, max_x, max_y
