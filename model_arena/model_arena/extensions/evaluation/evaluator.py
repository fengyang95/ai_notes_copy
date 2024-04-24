import re
import time
import pandas as pd

from tqdm.auto import tqdm
from abc import abstractmethod

from ..base import BaseExtension

from typing import Callable
from pandas.core.series import Series
from pandas.core.frame import DataFrame

DEFAULT_PROMPT = """You are a fair judge to decide which answer is better.
You should assign an integer score to each answer, with a perfect answer will have a score of 4.
The difference of scores MUST reflect your true preference.

The Question is:
{instruction}

The first answer is:
{output_y}

The second answer is:
{output_x}

**DO NOT** explain the reason.
ONLY return a tuple of two scores, with the following format: (INT, INT)
"""


class BaseEvaluator(BaseExtension):
    extension_name: str = "base"

    show_progress: bool = False

    def __init__(
        self,
        ignore: Callable | None = None,
        transform: Callable | None = None,
        show_progress: bool = False,
        **kwargs: dict[str, object],
    ) -> None:
        kwargs.update({"show_progress": show_progress})
        super().__init__(ignore, transform, **kwargs)

        tqdm.pandas(disable=not self.show_progress)

    @abstractmethod
    def _evaluate(self, df: DataFrame) -> DataFrame:
        ...

    def evaluate(self, df: DataFrame) -> DataFrame:
        # preprocess dataframe with _ignore and _transform
        df = self._preprocess(df)
        # evaluate
        df.loc[:, "auditor"] = self.extension_name
        df = self._evaluate(df)

        return df


class UnaryEvaluator(BaseEvaluator):
    extension_method: str = "unary"

    ignore_columns: list[str] = ["output"]
    transform_columns: list[str] = ["output"]


class PairwiseEvaluator(BaseEvaluator):
    extension_method: str = "pairwise"

    ignore_columns: list[str] = ["output_x", "output_y"]
    transform_columns: list[str] = ["output_x", "output_y"]


class ExactMatchEvaluator(UnaryEvaluator):
    extension_name: str = "exact_match"

    def _evaluate(self, df: DataFrame) -> DataFrame:
        df["score"] = df.progress_apply(lambda x: int(x["output"] == x["label"]), axis=1)
        return df


class ChatGPTEvaluator(PairwiseEvaluator):
    extension_name: str = "{model}"

    def __init__(
        self,
        ignore: Callable | None = None,
        transform: Callable | None = None,
        show_progress: bool = False,
        model: str = "gpt-4-0613",
        prompt: str | None = None,
        num_retries: int = 3,
    ) -> None:
        from ...core import BytedChatGPT

        super().__init__(
            ignore,
            transform,
            show_progress,
            model=model,
            prompt=prompt if prompt else DEFAULT_PROMPT,
            num_retries=num_retries,
        )
        self.llm = BytedChatGPT(model=self.model)
        self.extension_name = self.extension_name.format(model=self.model)

    def _evaluate_once(self, row: Series) -> Series:
        for _ in range(self.num_retries):
            # avoid high qps for gpt requests
            time.sleep(2)
            try:
                results = self.llm.invoke(
                    self.prompt.format(
                        instruction=row["instruction"],
                        output_x=row["output_x"],
                        output_y=row["output_y"],
                    )
                )
                if scores := re.search(r"\(\d,[ ]?\d\)", results)[0]:
                    return pd.Series(
                        tuple(map(int, scores[1:-1].replace(", ", ",").split(","))),
                        index=["score_x", "score_y"],
                    )
                else:
                    continue
            except Exception:
                continue

        return pd.Series((-1, -1), index=["score_x", "score_y"])

    def _evaluate(self, df: DataFrame) -> DataFrame:
        df[["score_x", "score_y"]] = df.progress_apply(self._evaluate_once, axis=1)
        return df
