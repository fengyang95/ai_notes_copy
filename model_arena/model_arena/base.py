import numpy as np
import pandas as pd
import numpy.typing as npt

from abc import ABC, abstractmethod
from sqlalchemy import select, delete, MetaData, Table

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

    def _check(self, x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        return np.isin(x, y)

    def _load_table(self, table_name: str) -> Table:
        table = Table(table_name, self.metadata, autoload_with=self.engine)
        return table

    def _execute(self, stmt: Executable) -> None:
        with self.engine.connect() as conn:
            conn.execute(stmt)
            conn.commit()

    def _dump(self, df: DataFrame, table: Table) -> None:
        table = str(table.name)
        df.to_sql(name=table, con=self.engine, if_exists="append", index=False)


class BaseModule(Base):
    meta_name: str
    meta_id: str
    meta_table: Table

    def _preload(self) -> None:
        table_name = f"{self.__class__.__name__.lower()}_meta"
        # load meta table
        self.meta_table = self._load_table(table_name)

    def __init__(self, engine: Engine) -> None:
        super().__init__(engine)

    @property
    def meta(self) -> DataFrame:
        # load all meta information from meta_table
        meta_stmt = select(self.meta_table)
        meta = pd.read_sql(meta_stmt, con=self.engine)
        if "id" in meta.columns:
            meta = meta.drop(columns=["id"])
        return meta

    @property
    def meta_names(self) -> npt.NDArray[np.str_]:
        # load all meta_names from meta_table
        meta_stmt = select(self.meta_table.c[self.meta_name])
        meta_names = pd.read_sql(meta_stmt, con=self.engine)[self.meta_name].values
        return meta_names

    def check(self, names: str | npt.ArrayLike) -> npt.NDArray[np.str_]:
        if names == "all":
            # return all meta_names
            return self.meta_names

        # convert names into npt.ArrayLike object
        if isinstance(names, str):
            names = [names]
        names = np.array(names)
        names_checked = self._check(names, self.meta_names)

        # some of the provided names are not found in meta_names
        if not names_checked.all():
            raise ValueError(
                f"Names {names[~names_checked]} not found in meta information.",
            )

        return names

    def _add_meta(self, name: str, records: dict[str, object]) -> None:
        # add a meta information into meta_table
        meta = pd.DataFrame.from_dict({name: records}, orient="index").reset_index(names=self.meta_name)
        self._dump(meta, table=self.meta_table)

    def _update_meta(self, name: str, records: dict[str, object]) -> None:
        # update a meta information into meta_table
        meta = pd.DataFrame.from_dict({name: records}, orient="index").reset_index(names=self.meta_name)
        self._dump(meta, table=self.meta_table)

    def _drop_meta(self, name: str) -> None:
        # drop a meta information from meta_table
        drop_meta_stmt = delete(self.meta_table).where(self.meta_table.c[self.meta_name] == name)
        self._execute(drop_meta_stmt)

    @abstractmethod
    def get(self, names: str | npt.ArrayLike) -> pd.DataFrame:
        ...

    @abstractmethod
    def add(self, *args) -> None:
        ...

    @abstractmethod
    def update(self, *args) -> None:
        ...

    @abstractmethod
    def drop(self, *args) -> None:
        ...
