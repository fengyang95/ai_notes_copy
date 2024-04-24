import uuid
import json
import pandas as pd

from sqlalchemy import select

from .base import BaseModule

from typing import Union, List, Dict, Any, Optional
from pandas.core.frame import DataFrame
from sqlalchemy.engine import Engine
from sqlalchemy.sql.schema import Table


class RawDatasets(BaseModule):
    meta_name: str = "dataset_name"
    meta_id: str = "dataset_id"

    raw_datasets_table: Table

    def _preload(self) -> None:
        super()._preload()
        # load raw_datsets_table
        self.raw_datasets_table = self._load_table(table_name="rawdatasets")

    def __init__(self, engine: Engine) -> None:
        super().__init__(engine)

    def get(self, datasets: Union[str, List[str]]) -> DataFrame:
        datasets = self.check(datasets)
        stmt = select(self.raw_datasets_table).where(self.raw_datasets_table.c[self.meta_name].in_(datasets))
        df = pd.read_sql(stmt, con=self.engine)

        return df

    def update(self, dataset: str, df: DataFrame) -> None:
        if all(self._check([dataset], self.meta[self.meta_name])):
            raise ValueError(
                f"Duplicate name {dataset} found, please use another raw dataset name.",
            )

        # required columns
        required_columns = ["tag", "information"]
        if not all(self._check(required_columns, df.columns)):
            raise ValueError(
                f"Columns {required_columns} is required in dataframe.",
            )
        df = df[required_columns]

        # update data
        df.loc[:, self.meta_name] = dataset
        df.loc[:, self.meta_id] = [uuid.uuid4().hex for _ in range(df.shape[0])]

        # final columns
        final_columns = [self.meta_name, self.meta_id, "tag", "information"]
        df = df[final_columns]

        # insert data
        self._dump(df, table=self.raw_datasets_table)

        # update meta
        records = {"length": df.shape[0]}
        self.update_meta(dataset, records)


class Datasets(BaseModule):
    meta_name: str = "dataset_name"
    meta_id: str = "dataset_id"

    datasets_table: Table

    raw_datasets: RawDatasets

    def _preload(self) -> None:
        super()._preload()
        # load datasets_table
        self.datasets_table = self._load_table(table_name="datasets")
        # load raw_datasets
        self.raw_datasets = RawDatasets(self.engine)

    def __init__(self, engine: Engine) -> None:
        super().__init__(engine)

    def get(self, datasets: Union[str, List[str]]) -> DataFrame:
        datasets = self.check(datasets)
        stmt = select(self.datasets_table).where(self.datasets_table.c[self.meta_name].in_(datasets))
        df = pd.read_sql(stmt, con=self.engine)

        return df

    def update(self, dataset: str, records: Dict[str, Any]) -> None:
        if all(self._check([dataset], self.meta[self.meta_name])):
            raise ValueError(
                f"Duplicate name {dataset} found, please use another dataset name.",
            )

        raw_dataset = records["raw_dataset"]
        template = records["instruction_template"]

        raw_df = self.raw_datasets.get(raw_dataset)

        raw_df[self.meta_name] = dataset
        raw_df[self.meta_id] = [uuid.uuid4().hex for _ in range(raw_df.shape[0])]
        raw_df["instruction"] = raw_df["information"].apply(lambda x: template.format(**json.loads(x)))

        # final columns
        final_columns = [self.meta_name, self.meta_id, "tag", "instruction"]
        raw_df = raw_df[final_columns]

        # insert data
        self._dump(raw_df, table=self.datasets_table)

        # update records
        records["length"] = raw_df.shape[0]
        # update meta
        self.update_meta(dataset, records)
