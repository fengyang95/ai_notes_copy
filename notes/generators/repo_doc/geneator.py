#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : geneator.py
# @Author  : Jimmy
# @Time    : 3/22/24 1:49 AM
# @Desc    :
from notes.config.config_loader import ConfigLoader
from notes.core.preprocess import RepositoryProcessor
from notes.readmeai.builder import MarkdownBuilder


def repo_info_creator(conf: ConfigLoader) -> None:
    """

    :param conf:
    :return:
    """
    repo_path = conf.repo_save_path
    repo_processor = RepositoryProcessor(conf)
    repo_context = repo_processor.generate_contents(repo_path)
    repo_context = repo_processor.language_mapper(repo_context)
    dependencies = repo_processor.get_dependencies(repo_context)
    conf.dependencies = dependencies
    conf.file_context = repo_context
    conf.config.md.tree = MarkdownBuilder(conf, dependencies, (), repo_path).md_tree


if __name__ == '__main__':
    pass
