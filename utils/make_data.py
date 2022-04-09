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
):
    """Generate the data needed for the low_resource language task"""
    x_train = []
    for input, tag in zip(train_input, train_tags):
        x_train.append(input + [Vocab.TOK] + tag)

    x_validation = []
    for input, tag in zip(validate_input, validate_tags):
        x_validation.append(input + [Vocab.TOK] + tag)

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
