import pytest

from model_arena import ModelArena
from model_arena.extensions import (
    PairwiseScoreStatAnalyst,
    EloRatingAnalyst,
    NeedleInAHaystackAnalyst,
)

from pandas.core.frame import DataFrame


@pytest.mark.parametrize(
    "datasets,models,analyst",
    [
        ("all", "gpt-4-0613", PairwiseScoreStatAnalyst()),
        ("IDE164", "all", EloRatingAnalyst()),
        ("niah_v1", "deepseek-coder-6.7b-instruct", NeedleInAHaystackAnalyst()),
    ],
)
def test_analyst(datasets, models, analyst):
    ma = ModelArena()
    result = ma.analysis(datasets=datasets, models=models, analyst=analyst)

    assert isinstance(result, DataFrame)
