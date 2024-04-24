import numpy as np
import pandas as pd
import numpy.typing as npt

from abc import ABC, abstractmethod
from scipy.optimize import fsolve

from typing import Callable
from pandas.core.series import Series
from pandas.core.frame import DataFrame


class BaseAnalyst(ABC):
    analyst: str = "base"
    analysis_method: str

    transform: Callable | None = None

    def __init__(self, transform: Callable | None = None) -> None:
        self.transform = transform

    def _transform(self, outputs: Series) -> Series:
        if self.transform is None:
            return outputs

        outputs_transformed = outputs.apply(lambda x: self.transform(x))
        return outputs_transformed

    @abstractmethod
    def analysis(self, df: DataFrame) -> DataFrame:
        ...


class UnaryAnalyst(BaseAnalyst):
    ...


class PairwiseAnalyst(BaseAnalyst):
    analysis_method: str = "pairwise"

    score_x_columm: str = "score_x"
    score_y_column: str = "score_y"

    @abstractmethod
    def _analysis(self, df: DataFrame) -> DataFrame:
        ...

    def analysis(self, df: DataFrame) -> DataFrame:
        # transform outputs
        df[self.score_x_columm] = df[self.score_x_columm].apply(self._transform)
        df[self.score_y_column] = df[self.score_y_column].apply(self._transform)
        # analysis dataframe
        df = self._analysis(df)

        return df


class PairwiseScoreStatAnalyst(PairwiseAnalyst):
    analyst: str = "pairwise_score_stat"

    def __init__(
        self,
        transform: Callable | None = None,
        pass_score: float = 2.0,
        perfect_score: float = 4.0,
    ) -> None:
        super().__init__(transform)

        self.pass_score = pass_score
        self.perfect_score = perfect_score

    def _analysis(self, df: DataFrame) -> DataFrame:
        metrics = ["pass_rate", "perfect_rate", "mean"]

        # group over dataset_id and tag
        df = (
            df.groupby(
                by=["dataset_id", "tag"],
            )[[self.score_x_columm, self.score_y_column]]
            .mean()
            .reset_index()
        )

        # stat over tag
        def stat(gdf: DataFrame) -> Series:
            # avoid zero division
            if gdf.shape[0] == 0:
                return pd.Series()

            # stat calculation
            pass_rate = (gdf[self.score_x_columm] == self.pass_score).sum() / gdf.shape[0]
            perfect_rate = (gdf[self.score_x_columm] == self.perfect_score).sum() / gdf.shape[0]
            mean = (gdf[self.score_x_columm]).mean()
            # rounding
            pass_rate = round(pass_rate * 100, 2)
            perfect_rate = round(perfect_rate * 100, 2)
            mean = round(mean, 2)

            s = pd.Series([pass_rate, perfect_rate, mean], index=metrics)
            return s

        df = df.groupby(by=["tag"]).apply(stat, include_groups=False).reset_index()

        return df


class EloRatingAnalyst(PairwiseAnalyst):
    analyst: str = "elo_rating"

    def __init__(
        self,
        transform: Callable | None = None,
        shift: int = 1500,
        scale: int = 400,
    ) -> None:
        super().__init__(transform)

        self.shift = shift
        self.scale = scale

    def _analysis(self, df: DataFrame) -> DataFrame:

        models = pd.concat((df["model_name_x"], df["model_name_y"])).drop_duplicates().values
        models_dict = {model: i for i, model in enumerate(models)}

        n_models = models.shape[0]
        win_games = np.zeros(n_models, dtype=np.int_)
        pairwise_games = np.zeros((n_models, n_models), dtype=np.int_)

        for _, row in df.iterrows():
            model_name_x, model_name_y = row["model_name_x"], row["model_name_y"]
            score_x, score_y = row["score_x"], row["score_y"]

            if score_x == score_y:
                continue

            if score_x > score_y:
                win_games[models_dict[model_name_x]] += 1
            else:
                win_games[models_dict[model_name_y]] += 1

            pairwise_games[models_dict[model_name_x], models_dict[model_name_y]] += 1

        pairwise_games += pairwise_games.T

        def f(x: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
            ret = win_games
            expx = np.exp(x)
            expxy = expx[:, np.newaxis] + expx[np.newaxis, :]
            ret = ret - expx * np.sum(1 / expxy * pairwise_games, axis=1)
            ret[0] = x[0]
            return ret

        root = fsolve(f, np.zeros((n_models)), xtol=1e-8)

        ratings = [self.scale * x + self.shift for x in root]
        ratings = {model: {"rating": rating} for model, rating in zip(models_dict.keys(), ratings)}

        df = (
            pd.DataFrame.from_dict(ratings, orient="index")
            .sort_values(by="rating", ascending=False)
            .reset_index(names="model")
        )

        return df
