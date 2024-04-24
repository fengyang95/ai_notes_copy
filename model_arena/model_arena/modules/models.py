import uuid
import pandas as pd
import numpy.typing as npt

from sqlalchemy import select

from ..base import BaseModule

from pandas.core.frame import DataFrame
from sqlalchemy.sql.schema import Table


class Models(BaseModule):
    meta_name: str = "model_name"
    meta_id: str = "model_id"
    models_table: Table

    def _preload(self) -> None:
        super()._preload()
        # load models_table, it is the same as meta_table
        self.models_table = self.meta_table

    def get(self, models: str | npt.ArrayLike) -> DataFrame:
        models = self.check(models)
        stmt = select(self.models_table).where(self.models_table.c[self.meta_name].in_(models))
        df = pd.read_sql(stmt, con=self.engine)
        if "id" in df.columns:
            df = df.drop(columns=["id"])

        return df

    def add(self, model: str, records: dict[str, object]) -> None:
        if self._check([model], self.meta_names).all():
            raise ValueError(
                f"Duplicate model {model} found, please use another model name.",
            )
        # add meta
        records.update({self.meta_id: uuid.uuid4().hex})
        self._add_meta(model, records)

    def update(self, model: str, records: dict[str, object]) -> None:
        if not self._check([model], self.meta_names).all():
            raise ValueError(
                f"Model {model} not found, if you wish to add a new model, use `add` method.",
            )
        # update meta
        records.update({self.meta_id: self.get_model_id(model)})
        self._update_meta(model, records)

    def drop(self, model: str) -> None:
        if not self._check([model], self.meta_names).all():
            raise ValueError(
                f"Model {model} not found, please check model name.",
            )

        # drop meta
        self._drop_meta(model)

    def get_model_id(self, model: str) -> str:
        model_id_stmt = select(
            self.meta_table.c[self.meta_id],
        ).where(self.meta_table.c[self.meta_name] == model)
        model_id = pd.read_sql(model_id_stmt, con=self.engine)[self.meta_id].values[0]

        return model_id

    def get_model_path(self, model: str) -> str:
        model_path_stmt = select(
            self.meta_table.c["model_path"],
        ).where(self.meta_table.c[self.meta_name] == model)
        model_path = pd.read_sql(model_path_stmt, con=self.engine)["model_path"].values[0]

        return model_path

    def get_model_name(self, model_id: str) -> str:
        model_name_stmt = select(
            self.meta_table.c[self.meta_name],
        ).where(self.meta_table.c[self.meta_id] == model_id)
        model_name = pd.read_sql(model_name_stmt, con=self.engine)[self.meta_name].values[0]

        return model_name
