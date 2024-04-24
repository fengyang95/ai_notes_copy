import numpy as np
import pandas as pd

from scipy.optimize import fsolve
from sqlalchemy import select, and_, or_
from sqlalchemy.orm import aliased

from .base import Base
from .datasets import Datasets
from .models import Models

from typing import List, Union, Optional
from pandas.core.series import Series
from pandas.core.frame import DataFrame
from sqlalchemy.engine import Engine
from sqlalchemy.sql.schema import Table


class ModelArena(Base):
    inferences_table: Table
    matches_table: Table

    def _preload(self) -> None:
        # load modules
        self.datasets = Datasets(self.engine)
        self.models = Models(self.engine)
        # load tables
        self.datasets_table = self.datasets.datasets_table
        self.models_table = self.models.models_table
        self.inferences_table = self._load_table(table_name="inferences")
        self.matches_table = self._load_table(table_name="matches")

    def __init__(self, engine: Engine) -> None:
        super().__init__(engine)

    def get_inferences(
        self,
        models: Union[str, List[str]],
        datasets: Union[str, List[str]],
    ) -> DataFrame:
        models = self.models.check(models)
        datasets = self.datasets.check(datasets)

        stmt = (
            select(
                self.datasets_table.c[self.datasets.meta_name, "tag", "instruction"],
                self.models_table.c[self.models.meta_name],
                self.inferences_table.c["prompt", "output"],
            )
            .join(
                self.inferences_table,
                self.datasets_table.c[self.datasets.meta_id] == self.inferences_table.c[self.datasets.meta_id],
            )
            .join(
                self.models_table,
                self.inferences_table.c[self.models.meta_id] == self.models_table.c[self.models.meta_id],
            )
            .where(
                and_(
                    self.datasets_table.c[self.datasets.meta_name].in_(datasets),
                    self.models_table.c[self.models.meta_name].in_(models),
                )
            )
        )
        df = pd.read_sql(stmt, con=self.engine)

        return df

    def update_inferences(self, df: DataFrame) -> None:
        # required columns
        required_columns = [
            self.datasets.meta_name,
            self.datasets.meta_id,
            self.models.meta_id,
            "prompt",
            "output",
        ]
        if not all(self._check(required_columns, df.columns)):
            raise ValueError(
                f"Columns {required_columns} is required in dataframe.",
            )
        df = df[required_columns]
        # insert data
        self._dump(df, table=self.inferences_table)

    def generate_inferences(
        self,
        models: Union[str, List[str]],
        datasets: Union[str, List[str]],
    ) -> DataFrame:
        models = self.models.check(models)
        datasets = self.datasets.check(datasets)

        stmt = select(
            self.datasets_table.c[self.datasets.meta_name, self.datasets.meta_id, "instruction"],
            self.models_table.c[self.models.meta_name, self.models.meta_id, "prompt_template"],
        ).where(
            and_(
                self.datasets_table.c[self.datasets.meta_name].in_(datasets),
                self.models_table.c[self.models.meta_name].in_(models),
            )
        )
        df = pd.read_sql(stmt, con=self.engine)

        # translate prompt_template into prompt
        df["prompt"] = df.apply(
            lambda x: x["prompt_template"].format(instruction=x["instruction"]),
            axis=1,
        )
        df = df.drop(columns=["prompt_template"])

        return df

    def get_matches(
        self,
        models: Union[str, List[str]],
        datasets: Union[str, List[str]],
    ) -> DataFrame:
        models = self.models.check(models)
        datasets = self.datasets.check(datasets)

        models_table_x = aliased(self.models_table)
        models_table_y = aliased(self.models_table)

        stmt = (
            select(
                self.datasets_table.c[self.datasets.meta_name, "tag", "instruction"],
                models_table_x.c[self.models.meta_name].label(f"{self.models.meta_name}_x"),
                self.matches_table.c["model_score_x"],
                models_table_y.c[self.models.meta_name].label(f"{self.models.meta_name}_y"),
                self.matches_table.c["model_score_y"],
            )
            .join(
                self.matches_table,
                self.datasets_table.c[self.datasets.meta_id] == self.matches_table.c[self.datasets.meta_id],
            )
            .join(
                models_table_x,
                self.matches_table.c[f"{self.models.meta_id}_x"] == models_table_x.c[self.models.meta_id],
            )
            .join(
                models_table_y,
                self.matches_table.c[f"{self.models.meta_id}_y"] == models_table_y.c[self.models.meta_id],
            )
            .where(
                and_(
                    self.matches_table.c[self.datasets.meta_name].in_(datasets),
                    or_(
                        models_table_x.c[self.models.meta_name].in_(models),
                        models_table_y.c[self.models.meta_name].in_(models),
                    ),
                )
            )
        )
        df = pd.read_sql(stmt, con=self.engine)

        return df

    def update_matches(self, df: DataFrame) -> None:
        # required columns
        required_columns = [
            f"{self.datasets.meta_name}",
            f"{self.datasets.meta_id}",
            f"{self.models.meta_id}_x",
            "model_score_x",
            f"{self.models.meta_id}_y",
            "model_score_y",
        ]
        if not all(self._check(required_columns, df.columns)):
            raise ValueError(
                f"Columns {required_columns} is required in dataframe.",
            )
        df = df[required_columns]

        self._dump(df, table=self.matches_table)

    def generate_matches(
        self,
        model: str,
        datasets: Union[str, List[str]],
        target_model: Optional[str] = None,
        shuffle: Optional[bool] = True,
    ) -> DataFrame:
        model = self.models.check(model)[0]
        model_id = self.models.get_model_id(model)
        datasets = self.datasets.check(datasets)

        if target_model is not None:
            target_model = self.models.check(target_model)[0]
            target_model_id = self.models.get_model_id(target_model)

        inferences_table_x = aliased(self.inferences_table)
        inferences_table_y = aliased(self.inferences_table)

        # TODO: fancy match pairing techniques should be written here
        # Currently, it is just a dummy strategy
        if target_model is None:
            match_condition = inferences_table_y.c[self.models.meta_id] != model_id
        else:
            match_condition = inferences_table_y.c[self.models.meta_id] == target_model_id

        stmt = (
            select(
                self.datasets_table.c[self.datasets.meta_name, self.datasets.meta_id, "tag", "instruction"],
                inferences_table_x.c[self.models.meta_id].label(f"{self.models.meta_id}_x"),
                inferences_table_x.c["output"].label("model_output_x"),
                inferences_table_y.c[self.models.meta_id].label(f"{self.models.meta_id}_y"),
                inferences_table_y.c["output"].label("model_output_y"),
            )
            .join(
                inferences_table_x,
                self.datasets_table.c[self.datasets.meta_id] == inferences_table_x.c[self.datasets.meta_id],
            )
            .outerjoin(
                inferences_table_y,
                inferences_table_x.c[self.datasets.meta_id] == inferences_table_y.c[self.datasets.meta_id],
            )
            .where(
                and_(
                    self.datasets_table.c[self.datasets.meta_name].in_(datasets),
                    inferences_table_x.c[self.models.meta_id] == model_id,
                    match_condition,
                )
            )
        )
        df = pd.read_sql(stmt, con=self.engine)

        if shuffle:
            # shuffle matches
            middle_point = df.shape[0] // 2
            df_u, df_l = df.iloc[:middle_point, :], df.iloc[middle_point:, :]
            # switch model_x and model_y
            df_l = df_l.rename(
                columns={
                    f"{self.models.meta_id}_x": f"{self.models.meta_id}_y",
                    "model_output_x": "model_output_y",
                    f"{self.models.meta_id}_y": f"{self.models.meta_id}_x",
                    "model_output_y": "model_output_x",
                }
            )
            # concat two parts
            df = pd.concat((df_u, df_l))
            # shuffle rows
            df = df.sample(frac=1, replace=False, ignore_index=True)

        return df

    def get_score_stat(
        self,
        model: str,
        dataset: str,
        perfect_score: Optional[int] = 4,
        pass_score: Optional[int] = 2,
    ) -> DataFrame:
        model = self.models.check(model)[0]
        model_id = self.models.get_model_id(model)
        dataset = self.datasets.check(dataset)[0]

        stmt_x = (
            select(
                self.matches_table.c[self.datasets.meta_id],
                self.datasets_table.c["tag"],
                self.matches_table.c["model_score_x"].label("model_score"),
            )
            .join(
                self.datasets_table,
                self.matches_table.c[self.datasets.meta_id] == self.datasets_table.c[self.datasets.meta_id],
            )
            .where(
                and_(
                    self.matches_table.c[self.datasets.meta_name] == dataset,
                    self.matches_table.c[f"{self.models.meta_id}_x"] == model_id,
                )
            )
        )
        stmt_y = select(
            self.matches_table.c[self.datasets.meta_id],
            self.matches_table.c["model_score_y"].label("model_score"),
        ).where(
            self.matches_table.c[f"{self.models.meta_id}_y"] == model_id,
        )

        df_x = pd.read_sql(stmt_x, con=self.engine)
        df_y = pd.read_sql(stmt_y, con=self.engine)
        df = pd.concat((df_x, df_y))

        # calculate over dataset_id
        df = df.groupby(by=[self.datasets.meta_id, "tag"]).mean().reset_index()

        # stat over tag
        def stat(gdf: DataFrame) -> Series:
            # no zero division
            if gdf.shape == 0:
                perfect_rate, pass_rate = 0, 0
                me = 0
            else:
                perfect_rate = (gdf["model_score"] == perfect_score).sum() / gdf.shape[0]
                pass_rate = (gdf["model_score"] >= pass_score).sum() / gdf.shape[0]
                me = (gdf["model_score"]).mean()

            perfect_rate = round(perfect_rate * 100, 2)
            pass_rate = round(pass_rate * 100, 2)

            s = pd.Series([perfect_rate, pass_rate, me], index=["perfect_rate", "pass_rate", "mean"])
            return s

        df = df.groupby(by=["tag"]).apply(stat).reset_index()
        return df

    def get_rating(self, datasets: Union[str, List[str]]) -> DataFrame:
        datasets = self.datasets.check(datasets)

        models_table_x = aliased(self.models_table)
        models_table_y = aliased(self.models_table)

        stmt = (
            select(
                self.matches_table.c[self.datasets.meta_name, "model_score_x", "model_score_y"],
                models_table_x.c[self.models.meta_name].label(f"{self.models.meta_name}_x"),
                models_table_y.c[self.models.meta_name].label(f"{self.models.meta_name}_y"),
            )
            .join(
                models_table_x,
                self.matches_table.c[f"{self.models.meta_id}_x"] == models_table_x.c[self.models.meta_id],
            )
            .join(
                models_table_y,
                self.matches_table.c[f"{self.models.meta_id}_y"] == models_table_y.c[self.models.meta_id],
            )
            .where(
                self.matches_table.c[self.datasets.meta_name].in_(datasets),
            )
        )
        df = pd.read_sql(stmt, con=self.engine)

        models = (
            pd.concat(
                (
                    df[f"{self.models.meta_name}_x"],
                    df[f"{self.models.meta_name}_y"],
                ),
                ignore_index=True,
            )
            .drop_duplicates()
            .to_list()
        )
        models = {model: i for i, model in enumerate(models)}

        n_models = len(models)
        win_games = np.zeros(n_models, dtype=np.int_)
        pairwise_games = np.zeros((n_models, n_models), dtype=np.int_)

        for _, row in df.iterrows():
            model_name_x, model_name_y = row[f"{self.models.meta_name}_x"], row[f"{self.models.meta_name}_y"]
            model_score_x, model_score_y = row["model_score_x"], row["model_score_y"]

            if model_score_x == model_score_y:
                continue

            if model_score_x > model_score_y:
                win_games[models[model_name_x]] += 1
            else:
                win_games[models[model_name_y]] += 1

            pairwise_games[models[model_name_x], models[model_name_y]] += 1

        pairwise_games += pairwise_games.T

        def f(x):
            ret = win_games
            expx = np.exp(x)
            expxy = expx[:, np.newaxis] + expx[np.newaxis, :]
            ret = ret - expx * np.sum(1 / expxy * pairwise_games, axis=1)
            ret[0] = x[0]
            return ret

        root = fsolve(f, np.zeros((n_models)), xtol=1e-8)

        scale = 400
        shift = 1500
        ratings = [scale * x + shift for x in root]
        ratings = {model_name: {"rating": rating} for model_name, rating in zip(models.keys(), ratings)}

        df = (
            pd.DataFrame.from_dict(ratings, orient="index")
            .sort_values(by="rating", ascending=False)
            .reset_index(names="model_name")
        )

        return df
