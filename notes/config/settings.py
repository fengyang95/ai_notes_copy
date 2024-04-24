#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : settings.py
# @Author  : Jimmy
# @Time    : 3/8/24 10:49 AM
# @Desc    :

"""
Data models and configuration settings for the readme-ai package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field, HttpUrl, validator
from dataclasses import dataclass

from notes.config.validators import GitValidator


class MarkdownSettings(BaseModel):
    """Markdown template blocks for the README.md file."""

    alignment: str
    badge_color: str
    badge_style: str
    badge_icons: str
    contribute: str
    emojis: bool
    features: str
    header: str
    image: str
    modules: str
    modules_widget: str
    overview: str
    placeholder: str
    quickstart: str
    shields_icons: str
    skill_icons: str
    slogan: str
    tables: str
    toc: str
    tree: str
    tree_depth: int
    width: str


class APISettings(BaseModel):
    """
    Universal LLM API settings.
    """

    content: Optional[str]
    rate_limit: Optional[int]


class FileSettings(BaseModel):
    """File paths used by the readme-ai CLI tool."""

    blacklist: str
    commands: str
    languages: str
    markdown: str
    parsers: str
    prompts: str
    shields_icons: str
    skill_icons: str


class GitSettings(BaseModel):
    """User repository settings, sanitized and validated by Pydantic."""

    repository: Union[str, Path] = Field(
        ...,  # default 省略号 Ellipsis
        description="The URL or directory path to the repository.",
    )
    full_name: Optional[str] = Field(
        None, description="The full name of the repository."
    )
    host_domain: Optional[str] = Field(
        None, description="The domain of the repository host."
    )
    host: Optional[str] = Field(None, description="The repository host.")
    name: Optional[str] = Field(
        None, description="The name of the repository."
    )

    _validate_repository = validator("repository", pre=True, always=True)(
        GitValidator.validate_repository  # 这个地方可以用来解析
    )
    _validate_full_name = validator("full_name", pre=True, always=True)(
        GitValidator.validate_full_name
    )
    _set_host_domain = validator("host_domain", pre=True, always=True)(
        GitValidator.set_host_domain
    )
    _set_host = validator("host", pre=True, always=True)(GitValidator.set_host)
    _set_name = validator("name", pre=True, always=True)(GitValidator.set_name)


class ModelSettings(BaseModel):
    """LLM API settings used for generating text for the README.md file."""

    api: Optional[str]
    base_url: Optional[HttpUrl]
    context_window: Optional[int]
    encoder: Optional[str]
    model: Optional[str]
    temperature: Optional[float]
    tokens: Optional[int]
    top_p: Optional[float]


class Settings(BaseModel):
    """Nested data model to store all configuration settings."""

    api: APISettings
    files: FileSettings
    git: GitSettings
    llm: ModelSettings

    class Config:
        """Pydantic configuration settings."""

        validate_assignment = True


@dataclass
class Request4AD:
    pass


@dataclass
class StatusInOut:
    pass


if __name__ == '__main__':
    repository = "https://github.com/codefuse-ai/codefuse-devops-eval"
    git_info = GitSettings(repository=repository)
    pass
