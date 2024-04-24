import json
import math
import numpy as np
import pandas as pd
import numpy.typing as npt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from abc import abstractmethod
from scipy.optimize import fsolve

from ..base import BaseExtension

from typing import Callable
from pandas.core.series import Series
from pandas.core.frame import DataFrame


class BaseAnalyst(BaseExtension):
    extension_name: str = "base"

    show_plot: bool = False
    plot_kwargs: dict[str, object] = ...

    def __init__(
        self,
        ignore: Callable | None = None,
        transform: Callable | None = None,
        show_plot: bool = False,
        plot_kwargs: dict[str, object] = ...,
        **kwargs: dict[str, object],
    ) -> None:
        kwargs.update({"show_plot": show_plot, "plot_kwargs": plot_kwargs})
        super().__init__(ignore, transform, **kwargs)

    @abstractmethod
    def _plot(self, df: DataFrame, **kwargs: dict[str, object]) -> None:
        ...

    @abstractmethod
    def _analysis(self, df: DataFrame) -> DataFrame:
        ...

    def analysis(self, df: DataFrame) -> DataFrame:
        # preprocess dataframe with _ignore and _transform
        df = self._preprocess(df)
        # analysis
        df = self._analysis(df)
        # plot analysis result
        if self.show_plot:
            self._plot(df, **self.plot_kwargs)

        return df


class UnaryAnalyst(BaseAnalyst):
    extension_method: str = "unary"

    ignore_columns: list[str] = ["score"]
    transform_columns: list[str] = ["score"]


class PairwiseAnalyst(BaseAnalyst):
    extension_method: str = "pairwise"

    ignore_columns: list[str] = ["score_x", "score_y"]
    transform_columns: list[str] = ["score_x", "score_y"]


class NeedleInAHaystackAnalyst(UnaryAnalyst):
    extension_name: str = "needle_in_a_haystack"

    def __init__(
        self,
        ignore: Callable | None = None,
        transform: Callable | None = None,
        show_plot: bool = False,
        plot_kwargs: dict[str, object] = ...,
        tokenize: Callable = lambda x: len(x),
    ) -> None:
        super().__init__(
            ignore,
            transform,
            show_plot,
            plot_kwargs,
            tokenize=tokenize,
        )

    def _construct_cmap(self) -> mcolors.LinearSegmentedColormap:
        pos = [0, 0.4, 0.55, 0.65, 0.85, 1]
        colors = ["#EC7461", "#F1A257", "#F5C759", "#C3BF6F", "#8FD780", "#63D3A5"]

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "",
            list(zip(pos, colors)),
        )
        return cmap

    def _plot(self, df: DataFrame, threshold: float = 0.8, image: str = "analysis.png") -> None:
        # reshape dataframe
        df = df.pivot(index="pos", columns="ctx", values="rate")

        # plot
        ax = sns.heatmap(
            df,
            cmap=self._construct_cmap(),
            annot=df[df < threshold].fillna(""),
            fmt="",
            cbar_kws={"label": "Accuracy of Retrieval"},
        )
        ax.set(
            xlabel="Context length (unit: k)",
            ylabel="Needle Depth",
        )

        # save plot
        plt.savefig(image, dpi=800)

    def _analysis(self, df: DataFrame) -> DataFrame:
        idf: DataFrame = df["information"].apply(lambda x: pd.Series(json.loads(x)))
        idf["prefix_length"] = idf["prefix"].apply(self.tokenize)
        idf["postfix_length"] = idf["postfix"].apply(self.tokenize)
        idf["total_length"] = idf["prefix_length"] + idf["postfix_length"]

        # context length
        idf["ctx"] = idf["total_length"].apply(lambda x: x // 1000)
        # needle position
        idf["pos"] = (idf["prefix_length"] / idf["total_length"]).apply(
            lambda x: math.floor(x * 10) * 1.0 / 10,
        )

        # concat final dataframe
        df = pd.concat((idf[["ctx", "pos"]], df[["score"]]), axis=1)

        # stat over ctx, pos
        def stat(grp: DataFrame) -> Series:
            # avoid zero division
            if grp.shape[0] == 0:
                return pd.Series()

            rate = round(grp["score"].sum() / grp.shape[0], 2)
            return pd.Series([rate], index=["rate"])

        df = df.groupby(by=["ctx", "pos"]).apply(stat, include_groups=False).reset_index()

        return df


class PairwiseScoreStatAnalyst(PairwiseAnalyst):
    extension_name: str = "pairwise_score_stat"

    def __init__(
        self,
        ignore: Callable | None = None,
        transform: Callable | None = None,
        show_plot: bool = False,
        plot_kwargs: dict[str, object] = ...,
        failed_score: float = 0.0,
        pass_score: float = 2.0,
        perfect_score: float = 4.0,
    ) -> None:
        super().__init__(
            ignore,
            transform,
            show_plot,
            plot_kwargs,
            failed_score=failed_score,
            pass_score=pass_score,
            perfect_score=perfect_score,
        )

    def _plot(self, df: DataFrame, **kwargs: dict[str, object]) -> None:
        print("Nothing to plot for pairwise_score_stat analyst.")

    def _analysis(self, df: DataFrame) -> DataFrame:
        metrics = ["failed_rate", "pass_rate", "perfect_rate", "mean"]

        # group over dataset_id and tag
        df = df.groupby(by=["dataset_id", "tag"])[["score_x", "score_y"]].mean().reset_index()

        # stat over tag
        def stat(grp: DataFrame) -> Series:
            # avoid zero division
            if grp.shape[0] == 0:
                return pd.Series()

            # stat calculation
            failed_rate = (grp["score_x"] == self.failed_score).sum() / grp.shape[0]
            pass_rate = (grp["score_x"] >= self.pass_score).sum() / grp.shape[0]
            perfect_rate = (grp["score_x"] == self.perfect_score).sum() / grp.shape[0]
            mean = (grp["score_x"]).mean()
            # rounding
            failed_rate = round(failed_rate * 100, 2)
            pass_rate = round(pass_rate * 100, 2)
            perfect_rate = round(perfect_rate * 100, 2)
            mean = round(mean, 2)

            s = pd.Series([failed_rate, pass_rate, perfect_rate, mean], index=metrics)
            return s

        df = df.groupby(by=["tag"]).apply(stat, include_groups=False).reset_index()

        return df


class EloRatingAnalyst(PairwiseAnalyst):
    extension_name: str = "elo_rating"

    def __init__(
        self,
        ignore: Callable | None = None,
        transform: Callable | None = None,
        show_plot: bool = False,
        plot_kwargs: dict[str, object] = ...,
        shift: int = 1500,
        scale: int = 400,
    ) -> None:
        super().__init__(
            ignore,
            transform,
            show_plot,
            plot_kwargs,
            shift=shift,
            scale=scale,
        )

    def _plot(self, df: DataFrame, **kwargs: dict[str, object]) -> None:
        print("Nothing to plot for elo_rating analyst.")

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
