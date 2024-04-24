#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : get_raw_files.py
# @Author  : Jimmy
# @Time    : 3/21/24 3:02 PM
# @Desc    :
import json
from typing import Dict, List

from notes.config.config_loader import ConfigLoader


def file_content_assemble(file_top_level_info, clazz_info, method_info):
    """

    :param file_top_level_info:
    :param clazz_info:
    :param method_info:
    :return:
    """
    if file_top_level_info.size == 0:
        file_top_level_info = ''
    else:
        file_top_level_info = json.loads(file_top_level_info.iloc[0]["attributes"])[2][1][0]
    clazz_info_details = ''
    for k, v in clazz_info.items():
        clazz_info_details += "\n** Class Name: " + k.split(".")[-1] + ", Description of the class as follows:" + "**\n"
        clazz_info_details += v + "\n"
    method_info_details = ''
    for k, v in method_info.items():
        method_info_details += "\n** Method Name: " + k.split(".")[
            -1] + ", Description of the Method as follows:" + "**\n"
        method_info_details += v + "\n"

    base_prompt = """All information within the code documentation is as follows.:

Top Level Information, this section provides an overview of the code that is outside any class or function definitions. 
{}

Classes Information, this section describes each class defined within the file.
{}

Methods Information, this section details the standalone functions that are defined outside the context of a class.
{}
"""
    return base_prompt.format(file_top_level_info, clazz_info_details, method_info_details)


def get_filename_from_uri(input_uri, repo_name):
    """
    从uri中提取file name

    :param repo_name:
    :param input_uri:
    :return:
    """
    return input_uri.split(repo_name)[-1].strip("/")


def file_context_rebuilder(conf: ConfigLoader, target_files: List[str], clazz_docs: Dict[str, str],
                           func_docs: Dict[str, str]):
    """

    :param conf:
    :param target_files:
    :param clazz_docs:
    :param func_docs:
    :return:
    """
    code_k_g = conf.code_k_g
    repo_name = conf.config.git.name
    # 获得clazz和func的mapper
    df_entity, df_uri_meta, df_sqlite_sequence, df_relation, df_entity_alias = \
        code_k_g[0], code_k_g[1], code_k_g[2], code_k_g[3], code_k_g[4]
    df_relation["unique_start_entityID"] = df_relation['uriID'].astype(str) + "&@" + df_relation['startEntityID']
    df_relation["unique_end_entityID"] = df_relation["uriID"].astype(str) + "&@" + df_relation['endEntityID']
    clazz_func_df_relation = df_relation[
        (df_relation['type'] == 'methodToClass') & (df_relation['endName'].notna()) & (df_relation['endName'] != '')]
    clazz_func_mapper = clazz_func_df_relation.groupby('unique_end_entityID')['unique_start_entityID'].apply(
        list).to_dict()
    # clazz_summary_dict
    df_entity["unique_entityID"] = df_entity['uriID'].astype(str) + "&@" + df_entity['entityID']
    df_uri_meta["file_name"] = df_uri_meta["uri"].apply(lambda x: get_filename_from_uri(x, repo_name))
    uri_id_filename_mapper = dict(zip(df_uri_meta['file_name'], df_uri_meta['uriID']))
    file_details = {}
    for file_name in target_files:
        uriID = uri_id_filename_mapper.get(file_name)
        # clazz
        unique_clazz_entity_ids = df_entity[(df_entity['uriID'] == uriID) & (df_entity['type'] == "clazz")][
            "unique_entityID"].tolist()
        clazz_info_unique = {k: v for k, v in clazz_docs.items() if k in unique_clazz_entity_ids}
        clazz_info = {k.split("&@")[1]: v for k, v in clazz_docs.items() if k in unique_clazz_entity_ids}
        clazz_sub_funcs = []
        for k, v in clazz_info_unique.items():
            clazz_sub_funcs += clazz_func_mapper.get(k)
        # func
        unique_func_entity_ids = df_entity[(df_entity['uriID'] == uriID) & (df_entity['type'] == "method")][
            "unique_entityID"].tolist()
        func_info = {k.split("&@")[1]: v for k, v in func_docs.items() if
                     k not in clazz_sub_funcs and k in unique_func_entity_ids}
        # file_info
        file_top_level_info = df_entity[(df_entity["uriID"] == uriID) & (df_entity["type"] == "fileTopLevel")]
        file_details[file_name] = file_content_assemble(file_top_level_info, clazz_info, func_info)
    return file_details


if __name__ == '__main__':
    pass
