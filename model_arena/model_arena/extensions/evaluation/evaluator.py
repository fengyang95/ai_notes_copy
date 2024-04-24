import os
import re
import time
import json
import string
import random
import subprocess
import pandas as pd

from tqdm.auto import tqdm
from abc import abstractmethod

from ..base import BaseExtension
from ...utils import extract_code_snippets

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

    @abstractmethod
    def _post_evaluate(self, df: DataFrame) -> tuple[DataFrame, DataFrame]:
        ...

    def evaluate(self, df: DataFrame) -> tuple[DataFrame, DataFrame]:
        # preprocess dataframe with _ignore and _transform
        df = self._preprocess(df)
        # evaluate
        df.loc[:, "auditor"] = self.extension_name
        df = self._evaluate(df)
        # post evaluate, split success and failed
        success_df, failed_df = self._post_evaluate(df)

        return success_df, failed_df


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

    def _post_evaluate(self, df: DataFrame) -> tuple[DataFrame, DataFrame]:
        return df, pd.DataFrame(columns=df.columns)


class PythonExecutionEvaluator(UnaryEvaluator):
    extension_name: str = "python_execution"

    def _is_passed(self, instruction: str, output: str, test: str) -> bool:
        code = f"{instruction}{output}\n{test}"
        test_file = f"{''.join(random.choices(string.ascii_lowercase, k=10))}.py"

        with open(test_file, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            subprocess.run(["python", f"{test_file}"], check=True, capture_output=True, text=True, timeout=5)
            result = True
        except Exception:
            result = False

        os.remove(test_file)

        return result

    def _evaluate_once(self, row: Series) -> Series:
        instruction, output = row["instruction"], row["output"]
        test = json.loads(row["information"])["test"]

        # try to obtain the code solution if possible
        # we only find the first code snippet wrapped in
        # the markdown format
        code_output = extract_code_snippets(output)
        if len(code_output) > 0:
            try:
                # this should find the proper code solution
                # if not we may only extract the language successfully
                # not the code itself
                code_output = code_output[0][1]
            except Exception:
                # fall back to blank code output
                code_output = ""
        else:
            # we don not find any code output
            code_output = ""

        # HACK: this is a hack to deal with the different type of model
        # we do not know if the model is a continual writing model or a chat model
        # so we check both possible solution
        passed = (
            self._is_passed(instruction, output, test)
            or self._is_passed("", output, test)
            or self._is_passed(instruction, code_output, test)
            or self._is_passed("", code_output, test)
        )

        return pd.Series([int(passed)], index=["score"])

    def _evaluate(self, df: DataFrame) -> DataFrame:
        df["score"] = df.progress_apply(self._evaluate_once, axis=1)
        return df

    def _post_evaluate(self, df: DataFrame) -> tuple[DataFrame, DataFrame]:
        return df, pd.DataFrame(columns=df.columns)


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
        self.llm = BytedChatGPT(model_name=self.model)
        self.extension_name = self.extension_name.format(model=self.model)
        # we set a wait time to avoid high qps request
        self._time_elapsed = 2

    def _evaluate_once(self, row: Series) -> Series:
        for _ in range(self.num_retries):
            # time waited to send another request
            time.sleep(self._time_elapsed)
            try:
                results = self.llm.invoke(
                    self.prompt.format(
                        instruction=row["instruction"],
                        output_x=row["output_x"],
                        output_y=row["output_y"],
                    )
                )
                # we successfully send a gpt request
                # we reset the wait time
                self._time_elapsed = 2
                if scores := re.search(r"\(\d,[ ]?\d\)", results)[0]:
                    return pd.Series(
                        tuple(map(int, scores[1:-1].replace(", ", ",").split(","))),
                        index=["score_x", "score_y"],
                    )
                else:
                    continue
            except Exception:
                # we failed to send a gpt request, we make wait time longer
                self._time_elapsed += 5
                continue

        return pd.Series((-1, -1), index=["score_x", "score_y"])

    def _evaluate(self, df: DataFrame) -> DataFrame:
        df[["score_x", "score_y"]] = df.progress_apply(self._evaluate_once, axis=1)
        return df

    def _post_evaluate(self, df: DataFrame) -> tuple[DataFrame, DataFrame]:
        failed_pattern = (df["score_x"] == -1) & (df["score_y"] == -1)
        success_df = df[~failed_pattern]
        failed_df = df[failed_pattern]

        return success_df, failed_df
