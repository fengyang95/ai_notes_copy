import pandas as pd

from sqlalchemy import create_engine, select, delete, and_, or_
from sqlalchemy.orm import aliased

from .base import Base
from .modules.datasets import Datasets
from .modules.models import Models
from .core.engine import LLMEngine
from .extensions.evaluation import UnaryEvaluator, PairwiseEvaluator
from .extensions.analysis import BaseAnalyst

from pandas.core.frame import DataFrame
from sqlalchemy.engine import Engine
from sqlalchemy.sql.schema import Table
from sqlalchemy.sql import ColumnCollection


class ModelArena(Base):
    default_db_psm: str = "toutiao.mysql.model_arena_write"

    inferences_table: Table
    inferences_extra_columns: list[tuple[str, type]] = [("prompt", str), ("output", str)]

    evaluations_table: Table
    evaluations_extra_columns: list[tuple[str, type]] = [("auditor", str), ("score", float)]

    matches_table: Table
    matches_extra_columns: list[tuple[str, type]] = [("auditor", str), ("score_x", float), ("score_y", float)]

    def _preload(self) -> None:
        # load modules
        self.datasets = Datasets(self.engine)
        self.models = Models(self.engine)
        # load tables
        self.raw_datasets_table = self.datasets.raw_datasets.raw_datasets_table
        self.datasets_table = self.datasets.datasets_table
        self.models_table = self.models.models_table
        self.inferences_table = self._load_table(table_name="inferences")
        self.matches_table = self._load_table(table_name="matches")
        self.evaluations_table = self._load_table(table_name="evaluations")

    def _create_default_engine(self) -> Engine:
        try:
            import bytedmysql
        except ImportError as e:
            print(
                "ModelArena default database is hosted on RDS in bytedance. "
                "To use it, you have to install bytedmysql by:\n"
                "pip install -i https://bytedpypi.byted.org/simple/ bytedmysql\n"
            )
            raise e

        uri = f"mysql+bytedmysql://:@/?db_psm={self.default_db_psm}"
        engine = create_engine(uri)
        return engine

    def __init__(self, engine: Engine | None = None) -> None:
        if engine is None:
            engine = self._create_default_engine()

        super().__init__(engine)

    @property
    def dataset_name(self) -> str:
        return self.datasets.meta_name

    @property
    def dataset_id(self) -> str:
        return self.datasets.meta_id

    @property
    def raw_dataset_id(self) -> str:
        return self.datasets.raw_meta_id

    @property
    def model_name(self) -> str:
        return self.models.meta_name

    @property
    def model_name_x(self) -> str:
        return f"{self.model_name}_x"

    @property
    def model_name_y(self) -> str:
        return f"{self.model_name}_y"

    @property
    def model_id(self) -> str:
        return self.models.meta_id

    @property
    def model_id_x(self) -> str:
        return f"{self.model_id}_x"

    @property
    def model_id_y(self) -> str:
        return f"{self.model_id}_y"

    @property
    def three_meta_tuple(self) -> list[str]:
        return [self.dataset_name, self.dataset_id, self.model_id]

    @property
    def four_meta_tuple(self) -> list[str]:
        return [self.dataset_name, self.dataset_id, f"{self.model_id}_x", f"{self.model_id}_y"]

    def _build_table_columns(self, table: Table, columns: list[str]) -> ColumnCollection:
        # XXX: this is a way to make code clearer when python version is below 3.11
        # since we can not use PEP646, this is a workaround of using
        # unpack in the subscripts
        # once we have bumped the minimal version of python to 3.11+
        # this function should be removed accordingly
        table_columns = ColumnCollection(
            columns=[table.c[column] for column in columns],
        )
        return table_columns

    def _build_extra_columns(self, df: DataFrame, extra_columns: list[tuple[str, type]]) -> None:
        for extra_column in extra_columns:
            df[extra_column[0]] = pd.Series(dtype=extra_column[1])

    def _generate_inferences(self, dataset: str, model: str) -> DataFrame:
        dataset = self.datasets.check(dataset)[0]
        model = self.models.check(model)[0]

        stmt = select(
            self.datasets_table.c[self.dataset_name, self.dataset_id, "instruction"],
            self.models_table.c[self.model_id, "prompt_template"],
        ).where(
            and_(
                self.datasets_table.c[self.dataset_name] == dataset,
                self.models_table.c[self.model_name] == model,
            )
        )
        df = pd.read_sql(stmt, con=self.engine)

        # build inferences extra columns
        self._build_extra_columns(df, self.inferences_extra_columns)

        # translate prompt_template into prompt
        df["prompt"] = df.apply(
            lambda x: x["prompt_template"].format(instruction=x["instruction"]),
            axis=1,
        )
        df = df.drop(columns=["instruction", "prompt_template"])

        return df

    def generate_inferences(self, dataset: str, model: str) -> DataFrame:
        print(
            f"You have directly call `generate_inferences` to acquire a dataframe, "
            f"this will build {self.inferences_extra_columns} columns in the dataframe. "
            f"Please remeber to fill them, before call `add_inferences`."
        )
        return self._generate_inferences(dataset, model)

    def add_inferences(self, df: DataFrame) -> None:
        required_columns = [*self.three_meta_tuple, *[c[0] for c in self.inferences_extra_columns]]
        if not self._check(required_columns, df.columns).all():
            raise ValueError(f"Columns {required_columns} is required in dataframe.")

        df = df[required_columns]
        self._dump(df, table=self.inferences_table)

    def get_inferences(self, datasets: str | list[str], models: str | list[str]) -> DataFrame:
        datasets = self.datasets.check(datasets)
        models = self.models.check(models)

        stmt = (
            select(
                self.datasets_table.c[self.dataset_name, self.dataset_id, "tag"],
                self.models_table.c[self.model_name],
                self.inferences_table.c["prompt", "output"],
            )
            .join(
                self.inferences_table,
                self.datasets_table.c[self.dataset_id] == self.inferences_table.c[self.dataset_id],
            )
            .join(
                self.models_table,
                self.inferences_table.c[self.model_id] == self.models_table.c[self.model_id],
            )
            .where(
                and_(
                    self.datasets_table.c[self.dataset_name].in_(datasets),
                    self.models_table.c[self.model_name].in_(models),
                )
            )
        )
        df = pd.read_sql(stmt, con=self.engine)

        return df

    def infer(self, dataset: str, model: str, engine: LLMEngine, upload: bool = True) -> DataFrame:
        # generate inference dataframe
        df = self._generate_inferences(dataset, model)
        # use llm engine to infer the result
        df = engine.infer(df)
        if upload:
            # add inference dataframe
            self.add_inferences(df)

        return df

    def _generate_evaluations(self, dataset: str, model: str) -> DataFrame:
        dataset = self.datasets.check(dataset)[0]
        model = self.models.check(model)[0]
        model_id = self.models.get_model_id(model)

        stmt = (
            select(
                self.datasets_table.c[self.dataset_name, self.dataset_id, "tag", "instruction"],
                self.datasets_table.c["output"].label("label"),
                self.raw_datasets_table.c["information"],
                self.inferences_table.c[self.model_id, "output"],
            )
            .join(
                self.raw_datasets_table,
                self.datasets_table.c[self.raw_dataset_id] == self.raw_datasets_table.c[self.dataset_id],
            )
            .join(
                self.inferences_table,
                self.datasets_table.c[self.dataset_id] == self.inferences_table.c[self.dataset_id],
            )
            .where(
                and_(
                    self.datasets_table.c[self.dataset_name] == dataset,
                    self.inferences_table.c[self.model_id] == model_id,
                )
            )
        )
        df = pd.read_sql(stmt, con=self.engine)

        # build evaluations extra columns
        self._build_extra_columns(df, self.evaluations_extra_columns)

        return df

    def generate_evaluations(self, dataset: str, model: str) -> DataFrame:
        print(
            f"You have directly call `generate_evaluations` to acquire a dataframe, "
            f"this will build {self.evaluations_extra_columns} columns in the dataframe. "
            f"Please remeber to fill them, before call `add_evaluations`."
        )
        return self._generate_evaluations(dataset, model)

    def add_evaluations(self, df: DataFrame) -> None:
        required_columns = [*self.three_meta_tuple, *[c[0] for c in self.evaluations_extra_columns]]
        if not self._check(required_columns, df.columns).all():
            raise ValueError(f"Columns {required_columns} is required in dataframe.")

        df = df[required_columns]
        self._dump(df, table=self.evaluations_table)

    def get_evaluations(self, datasets: str, models: str) -> DataFrame:
        datasets = self.datasets.check(datasets)
        models = self.models.check(models)

        stmt = (
            select(
                self.datasets_table.c[self.dataset_name, self.dataset_id, "tag", "instruction"],
                self.raw_datasets_table.c["information"],
                self.models_table.c[self.model_name],
                self.evaluations_table.c["auditor", "score"],
            )
            .join(
                self.raw_datasets_table,
                self.datasets_table.c[self.raw_dataset_id] == self.raw_datasets_table.c[self.dataset_id],
            )
            .join(
                self.evaluations_table,
                self.datasets_table.c[self.dataset_id] == self.evaluations_table.c[self.dataset_id],
            )
            .join(
                self.models_table,
                self.evaluations_table.c[self.model_id] == self.models_table.c[self.model_id],
            )
            .where(
                and_(
                    self.evaluations_table.c[self.dataset_name].in_(datasets),
                    self.models_table.c[self.model_name].in_(models),
                )
            )
        )
        df = pd.read_sql(stmt, con=self.engine)

        return df

    def evaluate(self, dataset: str, model: str, evaluator: UnaryEvaluator, upload: bool = True) -> DataFrame:
        assert evaluator.extension_method == "unary", "You have to use an unary evaluator."
        # generate evaluation dataframe
        df = self._generate_evaluations(dataset, model)
        # use evaluator to evaluate the result
        success_df, failed_df = evaluator.evaluate(df)
        if upload:
            # add evaluation dataframe
            self.add_evaluations(success_df)

        return success_df, failed_df

    def _match_pairing_technique(self, df: DataFrame, technique: str, target_model_id: str | None = None) -> DataFrame:
        # XXX: fancy match pairing technique coming!
        if technique == "all":
            df = df
        elif technique == "target":
            df = df[df[self.model_id_y] == target_model_id].reset_index(drop=True).copy()
        elif technique == "random":
            columns = [self.model_id_y, "output_y"]
            df = df.groupby(by=df.columns[~df.columns.isin(columns)].to_list()).sample(1).reset_index(drop=True).copy()
        else:
            print("Currently we do not support such pairing technique, return the full match dataframe.")
            df = df

        return df

    def _generate_matches(
        self,
        dataset: str,
        model: str,
        target_model: str | None = None,
        shuffle: bool = True,
    ) -> DataFrame:
        dataset = self.datasets.check(dataset)[0]
        model = self.models.check(model)[0]
        model_id = self.models.get_model_id(model)

        # setup match pairing technique
        if target_model is None:
            technique = "all"
            target_model_id = None
        elif target_model == "all":
            technique = "all"
            target_model_id = None
        elif target_model == "random":
            technique = "random"
            target_model_id = None
        else:
            technique = "target"
            target_model = self.models.check(target_model)[0]
            target_model_id = self.models.get_model_id(target_model)

        inferences_table_x, inferences_table_y = aliased(self.inferences_table), aliased(self.inferences_table)

        stmt = (
            select(
                self.datasets_table.c[self.dataset_name, self.dataset_id, "tag", "instruction"],
                inferences_table_x.c[self.model_id].label(self.model_id_x),
                inferences_table_x.c["output"].label("output_x"),
                inferences_table_y.c[self.model_id].label(self.model_id_y),
                inferences_table_y.c["output"].label("output_y"),
            )
            .join(
                inferences_table_x,
                self.datasets_table.c[self.dataset_id] == inferences_table_x.c[self.dataset_id],
            )
            .outerjoin(
                inferences_table_y,
                inferences_table_x.c[self.dataset_id] == inferences_table_y.c[self.dataset_id],
            )
            .where(
                and_(
                    self.datasets_table.c[self.dataset_name] == dataset,
                    inferences_table_x.c[self.model_id] == model_id,
                    inferences_table_y.c[self.model_id] != model_id,
                )
            )
        )
        df = pd.read_sql(stmt, con=self.engine)

        # apply match pairing technique
        df = self._match_pairing_technique(df, technique=technique, target_model_id=target_model_id)

        if shuffle:
            # shuffle matches
            middle_point = df.shape[0] // 2
            df_u, df_l = df.iloc[:middle_point, :], df.iloc[middle_point:, :]
            # switch model_x and model_y
            df_l = df_l.rename(
                columns={
                    self.model_id_x: self.model_id_y,
                    "output_x": "output_y",
                    self.model_id_y: self.model_id_x,
                    "output_y": "output_x",
                }
            )
            # concat two parts
            df = pd.concat((df_u, df_l))
            # shuffle rows
            df = df.sample(frac=1, replace=False, ignore_index=True)

        # build evaluations extra columns
        self._build_extra_columns(df, self.matches_extra_columns)

        return df

    def generate_matches(
        self,
        dataset: str,
        model: str,
        target_model: str | None = None,
        shuffle: bool = True,
    ) -> DataFrame:
        print(
            f"You have directly call `generate_matches` to acquire a dataframe, "
            f"this will build {self.matches_extra_columns} columns in the dataframe. "
            f"Please remeber to fill them, before call `add_matches`."
        )
        return self._generate_matches(dataset, model, target_model=target_model, shuffle=shuffle)

    def add_matches(self, df: DataFrame) -> None:
        required_columns = [*self.four_meta_tuple, *[c[0] for c in self.matches_extra_columns]]
        if not self._check(required_columns, df.columns).all():
            raise ValueError(f"Columns {required_columns} is required in dataframe.")

        df = df[required_columns]
        self._dump(df, table=self.matches_table)

    def match(
        self,
        dataset: str,
        model: str,
        evaluator: PairwiseEvaluator,
        target_model: str | None = None,
        upload: bool = True,
    ) -> DataFrame:
        assert PairwiseEvaluator.extension_method == "pairwise", "You have to use a pairwise evaluator."
        # generate match dataframe
        df = self._generate_matches(dataset, model, target_model, shuffle=True)
        # use evaluator to judge the preference
        success_df, failed_df = evaluator.evaluate(df)
        if upload:
            # add match dataframe
            self.add_matches(success_df)

        return success_df, failed_df

    def get_matches(
        self,
        datasets: str | list[str],
        models: str | list[str],
        target_models: str | None = None,
    ) -> DataFrame:
        datasets = self.datasets.check(datasets)
        models = self.models.check(models)

        if target_models is not None:
            target_models = self.models.check(target_models)

        models_table_x, models_table_y = aliased(self.models_table), aliased(self.models_table)

        # compose match condition
        if target_models is None:
            match_condition = and_(
                self.matches_table.c[self.dataset_name].in_(datasets),
                or_(
                    models_table_x.c[self.model_name].in_(models),
                    models_table_y.c[self.model_name].in_(models),
                ),
            )
        else:
            match_condition = and_(
                self.matches_table.c[self.dataset_name].in_(datasets),
                or_(
                    and_(
                        models_table_x.c[self.model_name].in_(models),
                        models_table_y.c[self.model_name].in_(target_models),
                    ),
                    and_(
                        models_table_x.c[self.model_name].in_(target_models),
                        models_table_y.c[self.model_name].in_(models),
                    ),
                ),
            )

        stmt = (
            select(
                self.datasets_table.c[self.dataset_name, self.dataset_id, "tag", "instruction"],
                self.raw_datasets_table.c["information"],
                models_table_x.c[self.model_name].label(self.model_name_x),
                models_table_y.c[self.model_name].label(self.model_name_y),
                self.matches_table.c["auditor", "score_x", "score_y"],
            )
            .join(
                self.raw_datasets_table,
                self.datasets_table.c[self.raw_dataset_id] == self.raw_datasets_table.c[self.dataset_id],
            )
            .join(
                self.matches_table,
                self.datasets_table.c[self.dataset_id] == self.matches_table.c[self.dataset_id],
            )
            .join(
                models_table_x,
                self.matches_table.c[self.model_id_x] == models_table_x.c[self.model_id],
            )
            .join(
                models_table_y,
                self.matches_table.c[self.model_id_y] == models_table_y.c[self.model_id],
            )
            .where(
                match_condition,
            )
        )
        df = pd.read_sql(stmt, con=self.engine)

        # reorder dataframe
        df_x = df[df[self.model_name_x].isin(models)]
        df_y = df[~df[self.model_name_x].isin(models)]
        # switch model_x and model_y
        df_y = df_y.rename(
            columns={
                self.model_name_x: self.model_name_y,
                "score_x": "score_y",
                self.model_name_y: self.model_name_x,
                "score_y": "score_x",
            }
        )
        df = pd.concat((df_x, df_y))

        return df

    def analysis(
        self,
        datasets: str | list[str],
        models: str | list[str],
        analyst: BaseAnalyst,
        target_models: str | list[str] | None = None,
    ) -> DataFrame:
        # get result dataframe
        if analyst.extension_method == "unary":
            df = self.get_evaluations(datasets, models)
        elif analyst.extension_method == "pairwise":
            df = self.get_matches(datasets, models, target_models=target_models)
        else:
            raise ValueError("Unknown analyst method, it should be one of [unary/pairwise].")

        # anlaysis!
        df = analyst.analysis(df)

        return df

    def _drop_dataset(self, dataset: str) -> None:
        self.datasets.drop(dataset)

        tables = [self.inferences_table, self.evaluations_table, self.matches_table]
        for table in tables:
            stmt = delete(table).where(table.c[self.dataset_name] == dataset)
            self._execute(stmt)

    def _drop_model(self, model: str) -> None:
        self.models.drop(model)

        model_id = self.models.get_model_id(model)
        tables = [self.inferences_table, self.evaluations_table, self.matches_table]
        for table in tables:
            stmt = delete(table).where(table.c[self.model_id] == model_id)
            self._execute(stmt)

    def drop(self, dataset: str | None = None, model: str | None = None, force: bool = False) -> None:
        if not force:
            raise ValueError(
                "Drop operation is dangerous, we highly recommend not to directly delete data.\n"
                "If you insist to delete data, you can set `force` to `True`."
            )

        if dataset is not None:
            self._drop_dataset(dataset)
        if model is not None:
            self._drop_model(model)
