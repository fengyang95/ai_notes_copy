import pandas as pd

from abc import ABC, abstractmethod
from sqlalchemy import select, delete, MetaData, Table

from typing import Union, List, Dict, Any
from pandas.core.series import Series
from pandas.core.frame import DataFrame
from sqlalchemy.engine import Engine
from sqlalchemy.sql.expression import Executable


class Base(ABC):
    base_dir: str

    @abstractmethod
    def _preload(self) -> None:
        ...

    def __init__(self, engine: Engine) -> None:
        # set up engine
        self.engine = engine
        self.metadata = MetaData()
        # preload
        self._preload()

    def _check(
        self,
        x: Union[Series, List[Any]],
        y: Union[Series, List[Any]],
    ) -> List[bool]:
        x_checked = pd.Series(x).isin(y).to_list()
        return x_checked

    def _load_table(self, table_name: str) -> Table:
        table = Table(table_name, self.metadata, autoload_with=self.engine)
        return table

    def _dump(self, df: DataFrame, table: Table) -> None:
        table = str(table.name)
        df.to_sql(name=table, con=self.engine, if_exists="append", index=False)

    def _execute(self, stmt: Executable) -> None:
        with self.engine.connect() as conn:
            conn.execute(stmt)
            conn.commit()


class BaseModule(Base):
    meta: pd.DataFrame
    meta_table: Table

    def _preload(self) -> None:
        table_name = f"{self.__class__.__name__.lower()}_meta"
        # load meta table
        self.meta_table = self._load_table(table_name)
        # load meta data
        stmt = select(self.meta_table)
        self.meta = pd.read_sql(stmt, con=self.engine)

    def __init__(self, engine: Engine) -> None:
        super().__init__(engine)

    def check(self, names: Union[str, List[str]]) -> List[str]:
        if names == "all":
            names = self.meta[self.meta_name].to_list()
        if isinstance(names, str):
            names = [names]

        names_check = self._check(names, self.meta[self.meta_name])

        if not all(names_check):
            raise ValueError(
                f"Names {names[~names_check]} not found in meta information.",
            )

        return names

    def update_meta(self, name: str, records: Dict[str, Any]) -> None:
        meta = pd.DataFrame.from_dict(
            {name: records},
            orient="index",
        ).reset_index(names=self.meta_name)

        self.meta = pd.concat((self.meta, meta))
        self._dump(self.meta, table=self.meta_table)

    def drop_meta(self, name: str) -> None:
        self.meta = self.meta[self.meta[self.meta_name] != name]

        drop_meta_stmt = delete(self.meta_table).where(self.meta_table.c[self.meta_name] == name)
        self._execute(drop_meta_stmt)

    @abstractmethod
    def get(self, names: Union[str, List[str]]) -> pd.DataFrame:
        ...

    @abstractmethod
    def update(self, *args) -> None:
        ...

    @abstractmethod
    def drop(self, *args) -> None:
        ...
