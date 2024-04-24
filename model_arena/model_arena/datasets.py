import uuid
import json
import numpy as np
import pandas as pd
import numpy.typing as npt
import pandas.api.types as ptypes

from sqlalchemy import select, delete

from .base import BaseModule

from pandas.core.frame import DataFrame
from sqlalchemy.sql.schema import Table


class RawDatasets(BaseModule):
    meta_name: str = "dataset_name"
    meta_id: str = "dataset_id"
    raw_datasets_table: Table

    def _preload(self) -> None:
        super()._preload()
        # load raw_datsets_table
        self.raw_datasets_table = self._load_table(table_name="rawdatasets")

    def get(self, datasets: str | npt.ArrayLike) -> DataFrame:
        datasets = self.check(datasets)
        stmt = select(self.raw_datasets_table).where(self.raw_datasets_table.c[self.meta_name].in_(datasets))
        df = pd.read_sql(stmt, con=self.engine)
        if "id" in df.columns:
            df = df.drop(columns=["id"])

        return df

    def add(self, dataset: str, df: DataFrame) -> None:
        if self._check([dataset], self.meta_names).all():
            raise ValueError(
                f"Duplicate raw dataset {dataset} found, please use another raw dataset name.",
            )

        # columns information
        required_columns = np.array(["tag", "information"])
        optional_columns = np.array(["output"])

        if not self._check(required_columns, df.columns).all():
            raise ValueError(
                f"Columns {required_columns} is required in dataframe.",
            )
        optional_columns = optional_columns[self._check(optional_columns, df.columns)]
        all_columns = np.concatenate((required_columns, optional_columns))
        df = df[all_columns]

        # convet informatin type
        if not ptypes.is_string_dtype(df["information"]):
            df["information"] = df["information"].apply(lambda x: json.dumps(x))

        # update data
        df.loc[:, self.meta_name] = dataset
        df.loc[:, self.meta_id] = [uuid.uuid4().hex for _ in range(df.shape[0])]

        # final columns
        final_columns = np.concatenate(([self.meta_name, self.meta_id], all_columns))
        df = df[final_columns]

        # add meta and data
        self._add_meta(dataset, records={"length": df.shape[0]})
        self._dump(df, table=self.raw_datasets_table)

    def update(self, dataset: str, df: DataFrame) -> None:
        raise NotImplementedError("Currently raw dataset does not support update method.")

    def drop(self, dataset: str) -> None:
        if not self._check([dataset], self.meta_names).all():
            raise ValueError(
                f"Name {dataset} not found, please check raw dataset name.",
            )

        # drop meta
        self._drop_meta(dataset)
        # drop data
        drop_data_stmt = delete(self.raw_datasets_table).where(self.raw_datasets_table.c[self.meta_name] == dataset)
        self._execute(drop_data_stmt)


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

    def get(self, datasets: str | npt.ArrayLike) -> DataFrame:
        datasets = self.check(datasets)
        stmt = select(self.datasets_table).where(self.datasets_table.c[self.meta_name].in_(datasets))
        df = pd.read_sql(stmt, con=self.engine)
        if "id" in df.columns:
            df = df.drop(columns=["id"])

        return df

    def add(self, dataset: str, records: dict[str, object]) -> None:
        if self._check([dataset], self.meta_names).all():
            raise ValueError(
                f"Duplicate dataset {dataset} found, please use another dataset name.",
            )

        # find raw_dataset and instruction_template in records
        raw_dataset_name = records["raw_dataset_name"]
        instruction_template = records["instruction_template"]

        # get raw dataset
        raw_df = self.raw_datasets.get(raw_dataset_name)

        # transforme raw dataset
        raw_df = raw_df.rename(columns={self.raw_datasets.meta_id: "raw_dataset_id"})
        raw_df["instruction"] = raw_df["information"].apply(lambda x: instruction_template.format(**json.loads(x)))

        # update raw dataset
        raw_df[self.meta_name] = dataset
        raw_df[self.meta_id] = [uuid.uuid4().hex for _ in range(raw_df.shape[0])]

        # final columns
        final_columns = np.array([self.meta_name, self.meta_id, "raw_dataset_id", "tag", "instruction", "output"])
        df = raw_df[final_columns]

        # update records
        records["length"] = df.shape[0]
        # add meta and data
        self._add_meta(dataset, records=records)
        self._dump(df, table=self.datasets_table)

    def update(self, dataset: str, records: dict[str, object]) -> None:
        raise NotImplementedError("Currently dataset does not support update method.")

    def drop(self, dataset: str) -> None:
        if not self._check([dataset], self.meta_names).all():
            raise ValueError(
                f"Name {dataset} not found, please check dataset name.",
            )

        # drop meta
        self._drop_meta(dataset)
        # drop data
        drop_data_stmt = delete(self.datasets_table).where(self.datasets_table.c[self.meta_name] == dataset)
        self._execute(drop_data_stmt)
