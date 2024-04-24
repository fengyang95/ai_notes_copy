#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : config_loader.py
# @Author  : Jimmy
# @Time    : 3/8/24 2:30 PM
# @Desc    :
import logging
from functools import cached_property
from pathlib import Path
from typing import Union, Optional

from pydantic import BaseModel

from notes.config.settings import APISettings, FileSettings, GitSettings, ModelSettings, MarkdownSettings
from notes.utils.file_handler import FileHandler
from notes.utils.file_resources import get_resource_path

_logger = logging.getLogger(__name__)


class Settings(BaseModel):
    """Nested data model to store all configuration settings."""

    api: APISettings  # done
    files: FileSettings
    git: GitSettings
    llm: ModelSettings
    md: MarkdownSettings

    class Config:
        """Pydantic configuration settings."""

        validate_assignment = True  # 表示要校验


class ConfigLoader:
    """Loads the configuration settings."""

    def __init__(
            self,
            config_file: Union[str, Path] = "config.toml",
            pacakge: Union[str, Path] = "config",
            sub_module: str = "settings",
    ) -> None:
        """Initialize ConfigLoader with the base configuration file."""
        self._logger = logging.getLogger(__name__)
        self.file_handler = FileHandler()
        self.config_file = config_file
        self.package = pacakge
        self.sub_module = sub_module
        self.config = self._base_config
        self.load_settings()
        # added by xujian
        self.code_k_g = Optional[list]
        self.dependencies = Optional[list]
        self.file_context = Optional[list]
        self.func_docs = Optional[dict]
        self.clazz_docs = Optional[dict]
        self.file_docs = Optional[dict]
        self.repo_save_path = Optional[str]
        self.doc_save_path = Optional[str]

    @cached_property
    def _base_config(self) -> Settings:
        """Loads the base configuration file.
        缓存属性，再一次执行的时候直接从缓存中读取
        """
        file_path = get_resource_path(file_path=self.config_file,
                                      package=self.package,
                                      sub_module=self.sub_module)
        config_dict = self.file_handler.read(file_path)
        return Settings.parse_obj(config_dict)

    def load_settings(self) -> dict[str, dict]:
        """Loads all configuration settings.

        - Loads the base configuration file from `settings/config.toml`.
        - Loads any additional configuration files specified in the base settings
          under the `files` key.

        Returns:
            A dictionary containing all loaded configuration settings, where
            the keys are the section names from `Settings` and the values
            are their respective data dictionaries.
        """
        settings = self._base_config.dict()  # 有缓存直接从缓存里取值

        for key, file_name in settings["files"].items():
            if not file_name.endswith(".toml"):
                continue
            file_path = get_resource_path(
                file_path=file_name,
                package=self.package,
                sub_module=self.sub_module
            )
            data_dict = self.file_handler.read(file_path)
            settings[key] = data_dict
            setattr(self, key, data_dict)  # 用来添加属性
            self._logger.info(
                f"Loaded configuration file: {self.sub_module}/{file_name}"
            )
        return settings


if __name__ == '__main__':
    config_loader = ConfigLoader()
    pass
