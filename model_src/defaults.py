"""Defaults for the model.

These values can be tweaked to affect model training performance.
"""


HIDDEN_UNITS = [64, 32]
LEARNING_RATE = 0.0001
BATCH_SIZE = 512
NUM_EPOCHS = 10
NUM_EVAL_STEPS = 100


def update_hyperparams(hyperparams: dict) -> dict:
    if "hidden_units" not in hyperparams:
        hyperparams["hidden_units"] = HIDDEN_UNITS
    else:
        if not isinstance(hyperparams['hidden_units'], list):
            hyperparams['hidden_units'] = [
                int(v) for v in hyperparams['hidden_units'].split(',')]
    if "learning_rate" not in hyperparams:
        hyperparams["learning_rate"] = LEARNING_RATE
    if "batch_size" not in hyperparams:
        hyperparams["batch_size"] = BATCH_SIZE
    if "num_epochs" not in hyperparams:
        hyperparams["num_epochs"] = NUM_EPOCHS
    return hyperparams