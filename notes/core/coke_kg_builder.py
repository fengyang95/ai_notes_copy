#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : coke_kg_builder.py
# @Author  : Jimmy
# @Time    : 3/18/24 11:25 AM
# @Desc    :
import sqlite3

import pandas as pd
import requests

API_URL = "http://api.example.com/generate_knowledge_graph"  # API接口URL


def query_code_k_g(repo_path, ckg_path) -> None:  # todo
    """

    :param repo_path: repo路径
    :param ckg_path: 输出路径
    :return:
    """
    # 输入配置信息和仓库路径
    payload = {
        'config': {
            'option1': repo_path,
            # Add more configuration options as needed
        },
        'repository_path': '/path/to/repository'
    }

    # 发送POST请求到API，传递配置信息和仓库路径
    response = requests.post(API_URL, json=payload)

    # 检查请求是否成功
    if response.status_code == 200:
        # 假设API响应内容是生成的SQLite3数据库文件的路径
        db_file_path = response.text  # 或者 response.json() 如果响应内容是JSON类型
    else:
        db_file_path = ''
        print(f"Failed to call API: {response.status_code} - {response.text}")
    # save 到 ckg_path
    print(ckg_path)


def read_all_table(db_path: str):
    """
    读取codeKg所有表

    :param db_path:
    :return:
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM uri_meta')
    uri_meta = cursor.fetchall()
    df_uri_meta = pd.DataFrame(uri_meta, columns=["uriID", "uri", "modifiedTime"])  # uri 表示文件路径/名
    cursor.execute('SELECT * FROM sqlite_sequence')
    sqlite_sequence = cursor.fetchall()
    df_sqlite_sequence = pd.DataFrame(sqlite_sequence, columns=["name", "seq"])
    cursor.execute('SELECT * FROM entity')
    entity = cursor.fetchall()
    df_entity = pd.DataFrame(entity, columns=["id", "uriID", "entityID", "type", "name", "attributes"])
    cursor.execute('SELECT * FROM relation')
    relation = cursor.fetchall()
    df_relation = pd.DataFrame(relation, columns=["id", "uriID", "startEntityID", "startName", "endEntityID", "endName",
                                                  "type", "attributes"])
    cursor.execute('SELECT * FROM entity_alias')
    entity_alias = cursor.fetchall()
    df_entity_alias = pd.DataFrame(entity_alias, columns=["id", "uriID", "entityID", "alias"])
    cursor.close()
    conn.close()

    return df_entity, df_uri_meta, df_sqlite_sequence, df_relation, df_entity_alias


if __name__ == '__main__':
    pass
