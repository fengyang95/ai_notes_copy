from abc import ABC, abstractmethod

from typing import Callable
from pandas.core.series import Series
from pandas.core.frame import DataFrame


class BaseEvaluator(ABC):
    evaluator: str = "base"
    evaluation_type: type
    evaluation_method: str

    transform: Callable | None = None
    ignore_output: str | None = None

    output_column: str = "output"
    label_column: str = "label"

    def __init__(self, transform: Callable | None = None, ignore_output: str | None = None) -> None:
        self.transform = transform
        self.ignore_output = ignore_output

    def _ignore(self, df: DataFrame) -> DataFrame:
        if self.ignore_output is None:
            return df

        df = df[df[self.output_column] != self.ignore_output]
        return df

    def _transform(self, outputs: Series) -> Series:
        if self.transform is None:
            return outputs

        outputs_transformed = outputs.apply(lambda x: self.transform(x))
        return outputs_transformed

    @abstractmethod
    def evaluate(self, df: DataFrame) -> DataFrame:
        ...


class UnaryEvaluator(BaseEvaluator):
    evaluation_method: str = "unary"

    @abstractmethod
    def _evaluate(self, outputs: Series, labels: Series) -> Series:
        ...

    def evaluate(self, df: DataFrame) -> DataFrame:
        # ignore outputs
        df = self._ignore(df)
        # transform outputs
        outputs: Series = self._transform(df[self.output_column])
        labels: Series = df[self.label_column]
        # calculate scores
        scores = self._evaluate(outputs, labels)
        # assign scores
        df.loc[:, "auditor"] = self.evaluator
        df.loc[:, "score"] = scores

        return df


class PairwiseEvaluator(BaseEvaluator):
    evaluation_method: str = "pairwise"


class ExactMatchEvaluator(UnaryEvaluator):
    evaluator: str = "exact_match"
    evaluation_type: type = bool

    def _evaluate(self, outputs: Series, labels: Series) -> Series:
        # calculate exact match scores
        scores: Series = (outputs == labels).astype(int)
        return scores
