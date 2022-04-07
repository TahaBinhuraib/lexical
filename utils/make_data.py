from data import encode


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
        x_train.append(input + tag)

    x_validation = []
    for input, tag in zip(validate_input, validate_tags):
        x_validation.append(input + tag)

    study = []
    for x, y in zip(x_train, train_output):
        study.append((x, y))

    test = []
    for x, y in zip(x_validation, validate_output):
        test.append((x, y))

    train_items, test_items = encode(study, vocab_x, vocab_y), encode(test, vocab_x, vocab_y)

    val_items = test_items

    return train_items, test_items, val_items, study, test

