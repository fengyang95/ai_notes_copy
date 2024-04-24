#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : get_func_docs.py
# @Author  : Jimmy
# @Time    : 3/13/24 11:20 AM
# @Desc    :
from pathlib import Path

import pandas as pd
import json

from typing import Dict

from tqdm import tqdm

from notes.config.config_loader import ConfigLoader
from notes._exceptions import LLMAPIServiceError
from notes.services.chatgpt import query_gpt

from notes.utils.file_handler import save_to_json, load_from_json

CLAZZ_DOC_FILE_NAME = "/clazz_doc_dict.json"


def creator(base_prompt, attributes, dependency_docs: Dict[str, str]):
    """

    :param base_prompt:
    :param attributes:
    :param dependency_docs:
    :return:
    """
    if len(attributes) != 3:
        clazz_comment = ''
    else:
        clazz_comment = attributes[0][1]  # 这里直接输入函数名即可
    func_info = "\n\n"
    for k, v in dependency_docs.items():
        func_info += "The Following is one of the class's methods API information\n" + v
    info = "The Comment of the class: \n" + clazz_comment + "\nThe information of the class's methods: \n" + func_info
    final_prompt = base_prompt.format(info)
    answer = query_gpt(final_prompt)
    # print(f"{answer}")
    if not answer:
        raise LLMAPIServiceError(f"LLM API response is None！", final_prompt)
    return answer


def clazz_doc_creator(conf: ConfigLoader) -> None:
    """

    :param conf:
    :return:
    """
    base_prompt = conf.prompts["prompts"]["clazz_doc"]
    code_k_g, docs_dir, func_docs = conf.code_k_g, conf.doc_save_path, conf.func_docs
    #
    df_entity, df_uri_meta, df_sqlite_sequence, df_relation, df_entity_alias = \
        code_k_g[0], code_k_g[1], code_k_g[2], code_k_g[3], code_k_g[4]
    df_merged = pd.merge(df_entity, df_uri_meta, left_on='uriID', right_on='uriID')
    df_merged["unique_entityID"] = df_merged['uriID'].astype(str) + "&@" + df_merged['entityID']
    # 处理func的调用关系
    df_relation["unique_start_entityID"] = df_relation['uriID'].astype(str) + "&@" + df_relation['startEntityID']
    df_relation["unique_end_entityID"] = df_relation["uriID"].astype(str) + "&@" + df_relation['endEntityID']
    clazz_func_df_relation = df_relation[
        (df_relation['type'] == 'methodToClass') & (df_relation['endName'].notna()) & (
                df_relation['endName'] != '')]
    grouped_df = clazz_func_df_relation.groupby('unique_end_entityID')

    temp_dir_path = Path(docs_dir)
    if not temp_dir_path.exists():
        temp_dir_path.mkdir(parents=True)

    clazz_docs = {}
    for unique_end_entityID, group in tqdm(grouped_df, desc="Clazz Docs In the process of generating..."):
        clazz_docs = load_from_json(docs_dir + CLAZZ_DOC_FILE_NAME)
        if unique_end_entityID in clazz_docs.keys():
            continue
        else:
            dependency_func_unique_entities = group["unique_start_entityID"].tolist()
            dependency_func_docs = {k: v for k, v in func_docs.items() if
                                    k in dependency_func_unique_entities}
            attributes = json.loads(
                df_merged[df_merged['unique_entityID'] == unique_end_entityID].iloc[0]["attributes"])
            clazz_docs[unique_end_entityID] = creator(base_prompt, attributes, dependency_func_docs)
            save_to_json(clazz_docs, docs_dir + CLAZZ_DOC_FILE_NAME)
    conf.clazz_docs = clazz_docs


if __name__ == '__main__':
    pass
