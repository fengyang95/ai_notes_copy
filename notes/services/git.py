#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : git.py
# @Author  : Jimmy
# @Time    : 3/7/24 9:41 PM
# @Desc    : Git operations for cloning and validating user repositories.

import os
import platform
import shutil
from enum import Enum
from pathlib import Path
from typing import Optional

import git
import logging

from notes._exceptions import GitCloneError

_logger = logging.getLogger(__name__)


class GitHost(str, Enum):
    """
    Enum class for Git service providers. Enum data includes the following:
        - domain name of the Git service
        - api url to fetch repository details
        - file url to format links in the generated README.md file
    """

    LOCAL = "local"
    GITHUB = "github.com"
    GITLAB = "gitlab.com"
    BITBUCKET = "bitbucket.org"

    # todo 看情况增加支持

    @property
    def api_url(self):
        """Gets the API URL for the Git service."""
        api_urls = {
            "local": None,
            "github.com": "https://api.github.com/repos/",
            "gitlab.com": "https://api.gitlab.com/v4/projects/",
            "bitbucket.org": "https://api.bitbucket.org/2.0/repositories/",
        }
        return api_urls[self.value]

    @property
    def file_url_template(self):
        """Gets the file URL template for accessing files on the Git service."""
        """blob确实有的，词汇“blob”原本在Git内部使用，用来指单个文件的内容"""
        file_url_templates = {
            "local": "{file_path}",
            "github.com": "https://github.com/{full_name}/blob/master/{file_path}",
            "gitlab.com": "https://gitlab.com/{full_name}/-/blob/master/{file_path}",
            "bitbucket.org": "https://bitbucket.org/{full_name}/notes/master/{file_path}",
        }
        return file_url_templates[self.value]


async def clone_repository(repository: str, temp_dir: str) -> str:
    """
    Clone repository to temporary directory and return the path.

    :param repository: URL of the remote Git repository to be cloned
    :param temp_dir: Path of the local directory where the cloned repository's code will be stored.
    :return: temp_dir, includes the cloned files.
    """
    try:
        temp_dir_path = Path(temp_dir)
        if not temp_dir_path.exists():
            temp_dir_path.mkdir(parents=True)
            git.Repo.clone_from(
                repository, temp_dir, depth=1, single_branch=True
            )
        if not os.listdir(temp_dir_path):  # todo 比较简单的处理
            git.Repo.clone_from(
                repository, temp_dir, depth=1, single_branch=True
            )
        if Path(repository).is_dir():  # 这里的流程是假如输入是本地仓库！
            repo = git.Repo.init(temp_dir)
            origin = repo.create_remote(
                "origin", f"file://{Path(repository).absolute()}"
            )
            repo.git.config("core.sparseCheckout", "true")

            sparse_checkout_path = (
                    temp_dir_path / ".git" / "info" / "sparse-checkout"
            )
            with sparse_checkout_path.open("w") as sc_file:
                sc_file.write("/*\n!.git/\n")

            origin.fetch()
            repo.git.checkout("FETCH_HEAD")

        await remove_hidden_contents(temp_dir_path)

        return temp_dir

    except (git.GitCommandError, git.InvalidGitRepositoryError, git.NoSuchPathError,) as exc:
        raise GitCloneError(f"Error cloning repository: {str(exc)}") from exc


async def remove_hidden_contents(directory: Path) -> None:
    """
    Remove hidden files and directories from a specified directory.

    :param directory:
    :return:
    """
    for item in directory.iterdir():
        if item.name.startswith(".") and item.name != ".github":
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


async def fetch_git_api_url(repo_url: str) -> str:
    """
    Parses the repository URL and returns the API URL.

    :param repo_url:
    :return:
    """
    # todo 确认作用是什么
    try:
        parts = repo_url.rstrip("/").split("/")
        repo_name = f"{parts[-2]}/{parts[-1]}"
        for service in GitHost:
            if service in repo_url:
                api_url = f"{service.api_url}{repo_name}"
                _logger.info(f"{service.name.upper()} API URL: {api_url}")
                return api_url

        raise ValueError("Unsupported Git service.")

    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid repository URL: {repo_url}") from exc


def fetch_git_file_url(file_path: str, full_name: str, repo_url: str) -> str:
    """
    Returns the URL of the file in the remote repository.

    :param file_path:
    :param full_name:
    :param repo_url:
    :return:
    """
    # todo 作用是什么
    if Path(repo_url).exists():
        return GitHost.LOCAL.file_url_template.format(file_path=file_path)

    for service in GitHost:
        if service in repo_url:
            return service.file_url_template.format(
                full_name=full_name, file_path=file_path
            )

    return file_path


def find_git_executable() -> Optional[Path]:
    """
    Find the path to the git executable, if available.

    :return:
    """
    # todo 啥意思
    try:
        git_exec_path = os.environ.get("GIT_PYTHON_GIT_EXECUTABLE")
        if git_exec_path:
            return Path(git_exec_path)

        # For Windows, set default location of git executable.
        if platform.system() == "Windows":
            default_windows_path = Path("C:\\Program Files\\Git\\cmd\\git.EXE")
            if default_windows_path.exists():
                return default_windows_path

        # For other OS, set executable path from PATH environment variable.
        paths = os.environ["PATH"].split(os.pathsep)
        for path in paths:
            git_path = Path(path) / "git"
            if git_path.exists():
                return git_path

        return None

    except Exception as exc:
        raise ValueError("Error finding Git executable") from exc


def validate_file_permissions(temp_dir: Path) -> None:
    """
    Validates file permissions of the cloned repository.

    :param temp_dir:
    :return:
    """
    # todo 啥意思？
    try:
        if platform.system() != "Windows":
            permissions = temp_dir.stat().st_mode & 0o777
            if permissions != 0o700:
                raise SystemExit(
                    f"Invalid file permissions for {temp_dir}.\n"
                    f"Expected 0o700, but found {oct(permissions)}."
                )

    except Exception as exc:
        raise ValueError(
            f"Error validating file permissions: {str(exc)}"
        ) from exc


def validate_git_executable(git_exec_path: str) -> None:
    """
    Validate the path to the git executable.

    :param git_exec_path:
    :return:
    """
    # todo 啥意思？
    try:
        if not git_exec_path or not Path(git_exec_path).exists():
            raise ValueError(f"Git executable not found at {git_exec_path}")

    except Exception as exc:
        raise ValueError("Error validating Git executable path") from exc


if __name__ == '__main__':
    pass
