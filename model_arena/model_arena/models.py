import uuid
import pandas as pd

from sqlalchemy import select

from .base import BaseModule

from typing import Union, List, Dict, Any
from pandas.core.frame import DataFrame
from sqlalchemy.engine import Engine
from sqlalchemy.sql.schema import Table


class Models(BaseModule):
    meta_name: str = "model_name"
    meta_id: str = "model_id"

    models_table: Table

    def _preload(self) -> None:
        super()._preload()
        # load models_table, it is the same as meta_table
        self.models_table = self.meta_table

    def __init__(self, engine: Engine) -> None:
        super().__init__(engine)

    def get(self, models: Union[str, List[str]]) -> DataFrame:
        models = self.check(models)
        stmt = select(self.models_table).where(self.models_table.c[self.meta_name].in_(models))
        df = pd.read_sql(stmt, con=self.engine)

        return df

    def update(self, model: str, records: Dict[str, Any]) -> None:
        if all(self._check([model], self.meta[self.meta_name])):
            raise ValueError(
                f"Duplicate name {model} found, please use another model name.",
            )

        # update meta
        records.update({self.meta_id: uuid.uuid4().hex})
        self.update_meta(model, records)

    def drop(self, model: str) -> None:
        if not all(self._check([model], self.meta[self.meta_name])):
            raise ValueError(
                f"Model {model} not found, please check model name.",
            )

        # drop meta
            self.drop_meta(model)

    def get_model_id(self, model: str) -> str:
        return self.meta[self.meta[self.meta_name] == model][self.meta_id].values[0]

    def get_model_name(self, model_id: str) -> str:
        return self.meta[self.meta[self.meta_id] == model_id][self.meta_name].values[0]
