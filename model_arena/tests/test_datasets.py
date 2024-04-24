import random
import string
import numpy as np
import pandas as pd

from model_arena import ModelArena

ma = ModelArena()


def test_datasets():
    raw_dataset_name = "".join(random.choices(string.ascii_lowercase, k=10))
    dataset_name = "".join(random.choices(string.ascii_lowercase, k=5))

    # demo dataframe
    df = pd.DataFrame(
        [["demo", {"component 1": "xxx", "component 2": "yyy"}]],
        columns=["tag", "information"],
    )

    # add raw dataset
    ma.datasets.raw_datasets.add(dataset=raw_dataset_name, df=df)
    assert np.isin([raw_dataset_name], ma.datasets.raw_datasets.meta_names).all()

    # add dataset
    ma.datasets.add(
        dataset=dataset_name,
        records={
            "raw_dataset_name": raw_dataset_name,
            "instruction_template": "Component 1: {component 1}, Component 2: {component 2}",
        },
    )
    assert np.isin([dataset_name], ma.datasets.meta_names).all()

    # drop dataset
    ma.datasets.drop(dataset=dataset_name)
    assert not np.isin([dataset_name], ma.datasets.meta_names)

    # drop raw dataset
    ma.datasets.raw_datasets.drop(dataset=raw_dataset_name)
    assert not np.isin([raw_dataset_name], ma.datasets.raw_datasets.meta_names).all()
