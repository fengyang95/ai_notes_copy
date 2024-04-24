#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : generator.py
# @Author  : Jimmy
# @Time    : 3/22/24 12:31 AM
# @Desc    :
from pathlib import Path

from tqdm import tqdm

from notes.config.config_loader import ConfigLoader
from notes._exceptions import LLMAPIServiceError
from notes.generators.file_doc.file_builder import file_context_rebuilder
from notes.services.chatgpt import query_gpt
from notes.utils.file_handler import save_to_json, load_from_json

FILE_SUMMARY_FILE_NAME = "/file_summary_dict.json"


def api_query(prompt):
    answer = query_gpt(prompt)
    if not answer:
        raise LLMAPIServiceError(f"LLM API response is None！", prompt)
    return answer


def file_docs_creator(conf: ConfigLoader):
    """

    :param conf:
    :return:
    """
    docs_dir, clazz_docs, func_docs = conf.doc_save_path, conf.clazz_docs, conf.func_docs
    file_name_list = [value.file_name for value in conf.file_context]
    file_details = file_context_rebuilder(conf, file_name_list, clazz_docs, func_docs)
    rebuild_file_context = [(k, v) for k, v in file_details.items()]
    conf.file_context = rebuild_file_context
    # 获取prompt
    prompts = conf.prompts
    # 获取repo tree
    tree = conf.config.md.tree
    #
    temp_dir_path = Path(docs_dir)
    if not temp_dir_path.exists():
        temp_dir_path.mkdir(parents=True)
    file_docs = {}
    for file_path, file_content in tqdm(rebuild_file_context, desc="File Summary In the process of generating..."):
        file_docs = load_from_json(docs_dir + FILE_SUMMARY_FILE_NAME)
        if file_path in file_docs.keys():
            continue
        else:
            prompt = prompts["prompts"]["file_summary"].format(
                tree, file_path, file_content)
            file_docs[file_path] = api_query(prompt)
            save_to_json(file_docs, docs_dir + FILE_SUMMARY_FILE_NAME)
    conf.file_docs = file_docs


if __name__ == '__main__':
    pass
