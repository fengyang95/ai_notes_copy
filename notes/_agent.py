#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : _agent.py
# @Author  : Jimmy
# @Time    : 3/7/24 7:58 PM
# @Desc    :
import asyncio
import os
import traceback
import logging
from pathlib import Path
from typing import Optional

from notes.config.config_loader import ConfigLoader
# from notes.config import ImageOptions
# from notes.config import GitSettings
from notes._exceptions import ReadmeGeneratorError
from notes.config.settings import GitSettings
from notes.core.coke_kg_builder import read_all_table, query_code_k_g
from notes.generators.class_doc.generator import clazz_doc_creator
from notes.generators.file_doc.generator import file_docs_creator
from notes.generators.func_doc.generator import func_doc_creator
from notes.generators.repo_doc.geneator import repo_info_creator
from notes.models.dalle import DalleHandler
from notes.models.factory import ModelFactory
from notes.readmeai.builder import MarkdownBuilder
from notes.services.git import clone_repository
from notes.utils.file_handler import FileHandler

_logger = logging.getLogger(__name__)

CODE_BASE_NAME = "ai_notes"
REPO_SAVE_PATH = CODE_BASE_NAME + "/examples/{repo_name}"
DOC_SAVE_PATH = CODE_BASE_NAME + "/examples/{repo_name}_docs"
CKG_FILE_NAME = "/ckg.sqlite3"


def agent(repository: str, ckg_path: Optional[str]) -> None:
    """
    Configures and runs the README file generator agent.

    :param repository: git仓库地址
    :param ckg_path: 假如为None，需要去请求CKG计算接口；不为none，直接读取ckg sqlite3 db
    :return:
    """
    output_file = ''
    try:
        conf = ConfigLoader()
        # get repository
        conf.config.git = GitSettings(repository=repository)
        asyncio.run(download_repository(conf))
        _logger.info(f"Repository validated: {conf.config.git}")

        # get code knowledge Graph
        asyncio.run(get_code_knowledge_graph(conf, ckg_path))
        _logger.info(f"CodeKnowledgeGraph validated: {len(conf.code_k_g)}")

        # get api docs
        asyncio.run(assets_doc_generator(conf))
        _logger.info(f"Repository validated: {conf.config.git}")

        # get readme
        asyncio.run(readme_generator(conf, output_file))
        _logger.info(f"Repository validated: {conf.config.git}")

    except Exception as exc:
        raise ReadmeGeneratorError(exc, traceback.format_exc()) from exc


async def download_repository(conf: ConfigLoader) -> None:
    """
    clone_repository

    :param conf:
    :return:
    """
    repo_dir = os.path.join(os.path.realpath(__file__).split(CODE_BASE_NAME)[0],
                            REPO_SAVE_PATH.format(repo_name=conf.config.git.name))
    conf.repo_save_path = repo_dir
    await clone_repository(conf.config.git.repository, repo_dir)


async def get_code_knowledge_graph(conf: ConfigLoader, ckg_path: str) -> None:
    """
    get code_knowledge_graph

    :param conf:
    :param ckg_path:
    :return:
    """
    conf.doc_save_path = os.path.join(os.path.realpath(__file__).split(CODE_BASE_NAME)[0],
                                      DOC_SAVE_PATH.format(repo_name=conf.config.git.name))
    if ckg_path:
        conf.code_k_g = read_all_table(ckg_path)
    else:
        ckg_path = conf.doc_save_path + CKG_FILE_NAME
        # 存入目标文件夹
        query_code_k_g(conf.repo_save_path, ckg_path)
    conf.code_k_g = read_all_table(db_path)


async def assets_doc_generator(conf: ConfigLoader) -> None:
    """

    :param conf:
    :return:
    """
    # get function docs
    func_doc_creator(conf)
    # get clazz docs
    clazz_doc_creator(conf)
    # get repo information for next feature create!
    repo_info_creator(conf)
    # get repository
    file_docs_creator(conf)


async def readme_generator(conf: ConfigLoader, output_file: Path) -> None:
    """Orchestrates the README.md file generation process."""
    temp_dir = repo_save_path(conf)
    dependencies = conf.dependencies
    raw_files = conf.file_context

    async with ModelFactory.model_handler(conf).use_api() as llm:
        responses = await llm.batch_request(dependencies, raw_files)
        (
            summaries,
            features,
            overview,
            slogan,
        ) = responses
        conf.config.md.features = conf.config.md.features.format(features)
        conf.config.md.overview = conf.config.md.overview.format(overview)
        conf.config.md.slogan = slogan

    if conf.config.md.image == ImageOptions.LLM.value:
        conf.config.md.width = "50%"
        dalle = DalleHandler(conf)
        image_url = dalle.run()
        conf.config.md.image = dalle.download(image_url)

    readme_md = MarkdownBuilder(
        conf, dependencies, summaries, temp_dir
    ).build()

    FileHandler().write(output_file, readme_md)
    _logger.info("README generation process completed successfully!")
    _logger.info(f"README.md file saved to: {output_file}")
    _logger.info("Share it @ github.com/eli64s/readme-ai/discussions")


def repo_save_path(conf, is_output=False):
    if not is_output:
        base_path = CODE_BASE_NAME + "/examples/" + conf.config.git.name
    else:
        base_path = CODE_BASE_NAME + "/examples/" + conf.config.git.name + "_docs"
    return os.path.join(os.path.realpath(__file__).split(CODE_BASE_NAME)[0], base_path)


if __name__ == '__main__':
    db_path = "/Users/bytedance/ai_notes/examples/dycause_rca_docs/dycause_rca.sqlite3"
    agent("https://github.com/PanYicheng/dycause_rca", ckg_path=db_path)
    pass
