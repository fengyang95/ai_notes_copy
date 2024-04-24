import random
import string
import numpy as np

from model_arena import ModelArena

ma = ModelArena()


def test_models():
    model_name = "".join(random.choices(string.ascii_lowercase, k=30))
    model_path = "".join(random.choices(string.ascii_lowercase, k=100))

    # add model
    ma.models.add(model=model_name, records={"model_path": model_path})
    assert np.isin([model_name], ma.models.meta_names)

    # drop model
    ma.models.drop(model=model_name)
    assert not np.isin([model_name], ma.models.meta_names)
