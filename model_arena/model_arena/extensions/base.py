from abc import ABC

from typing import Callable
from pandas.core.frame import DataFrame


class BaseExtension(ABC):
    extension_name: str
    extension_method: str

    ignore: Callable | None = None
    ignore_columns: list[str] = ...
    transform: Callable | None = None
    transform_columns: list[str] = ...

    def _check(self):
        if self.ignore is not None and len(self.ignore_columns) == 0:
            raise ValueError(
                "You have provided a ignore function, please also provide the columns to apply this function."
            )

        if self.transform is not None and len(self.transform_columns) == 0:
            raise ValueError(
                "You have provided a transform function, please also provide the columns to apply this function."
            )

    def _set_attr(self, **kwargs: dict[str, object]) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __init__(
        self,
        ignore: Callable | None = None,
        transform: Callable | None = None,
        **kwargs: dict[str, object],
    ) -> None:
        self.ignore = ignore
        self.transform = transform

        self._check()
        self._set_attr(**kwargs)

    def _ignore(self, df: DataFrame) -> DataFrame:
        if self.ignore is None:
            return df

        df = df[~df[self.ignore_columns].map(self.ignore).any(axis=1)].copy()
        return df

    def _transform(self, df: DataFrame) -> DataFrame:
        if self.transform is None:
            return df

        df[self.transform_columns] = df[self.transform_columns].map(self.transform)
        return df

    def _preprocess(self, df: DataFrame) -> DataFrame:
        df = self._ignore(df)
        df = self._transform(df)

        return df
