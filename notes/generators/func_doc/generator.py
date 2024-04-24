from pathlib import Path
from typing import Dict

import pandas as pd
import json
from collections import deque

from notes.config.config_loader import ConfigLoader
from notes._exceptions import LLMAPIServiceError
from tqdm import tqdm

from notes.services.chatgpt import query_gpt
from notes.utils.file_handler import save_to_json, load_from_json

FUNC_DOC_FILE_NAME = "/func_doc_dict.json"


def get_core_func_info(input_func_doc):
    """
    获取func的关键信息，具体后面可以调整

    :param input_func_doc:
    :return:
    """
    # Split the API documentation into lines
    lines = input_func_doc.split('\n')
    # Find the line with 'Parameters'
    parameters_index = next(i for i, line in enumerate(lines) if 'Parameters' in line)
    # Get the summary text
    function_summary = lines[0: parameters_index]  # 假如没找到，就全量也没啥大问题
    return function_summary


def creator(base_prompt, attributes, dependency_docs: Dict[str, str]):
    """
    函数doc生成prompt组装

    :param base_prompt:
    :param attributes:
    :param dependency_docs:
    :return:
    """
    if len(attributes) != 5:
        print(f"Formatting Error!!!{attributes}")
        raise Exception
    content_chunks = "".join(attributes[4][1])
    if len(dependency_docs) == 0:
        final_prompt = base_prompt.format(content_chunks, '')
    else:
        pre_fix = "The API documents of the referenced functions are as follows:"
        for k, v in dependency_docs.items():
            pre_fix += get_core_func_info(v) + "\n"
        final_prompt = base_prompt.format(content_chunks, pre_fix)
    answer = query_gpt(final_prompt)
    if not answer:
        raise LLMAPIServiceError(f"LLM API response is None！", final_prompt)
    return answer


def topological_sort(graph):
    """
    拓扑排序，确保在处理每个节点之前，该节点的依赖（即它的子节点）已经被处理过了。
    :param graph:
    :return:
    """
    in_degree = {node: 0 for node in graph.keys()}  # 初始化所有节点的入度为0
    for node in graph:
        for child in graph[node]:
            try:
                in_degree[child] += 1  # 对有入边的节点入度加1
            except Exception as e:
                print(in_degree)
                print(child)
                break

    # 入度为0的节点队列
    zero_in_degree_queue = deque([node for node in in_degree if in_degree[node] == 0])

    # 保存排序结果
    sorted_nodes = []

    while zero_in_degree_queue:
        node = zero_in_degree_queue.popleft()  # 取出一个入度为0的节点
        sorted_nodes.append(node)  # 放入排序结果中
        for child in graph[node]:
            in_degree[child] -= 1  # 将该节点的所有子节点的入度都减少1
            if in_degree[child] == 0:  # 如果子节点入度变为0，加入队列
                zero_in_degree_queue.append(child)

    # 检查是否存在环
    if len(sorted_nodes) != len(graph):
        raise Exception("Graph has at least one cycle, topological sort is not possible")

    # 对每个节点调用xx_operation
    return sorted_nodes


def func_doc_creator(conf: ConfigLoader) -> None:
    """
    生成函数级别描述文件

    :param conf:
    :return: None, 生成的内容按照json格式保存
    """
    base_prompt = conf.prompts["prompts"]["func_doc"]
    code_k_g, docs_dir = conf.code_k_g, conf.doc_save_path
    df_entity, df_uri_meta, df_sqlite_sequence, df_relation, df_entity_alias = \
        code_k_g[0], code_k_g[1], code_k_g[2], code_k_g[3], code_k_g[4]
    df_merged = pd.merge(df_entity, df_uri_meta, left_on='uriID', right_on='uriID')
    df_merged["unique_entityID"] = df_merged['uriID'].astype(str) + "&@" + df_merged['entityID']

    # 处理func的调用关系
    df_relation["unique_start_entityID"] = df_relation['uriID'].astype(str) + "&@" + df_relation['startEntityID']
    caller2callee_df_relation = df_relation[df_relation['type'] == "callerToCallee"]
    grouped_df = caller2callee_df_relation.groupby('unique_start_entityID')
    caller_to_callee = {}
    for unique_start_EntityID, group in grouped_df:
        # 因为存在很多endEntityName为空的情况，所以这里选用endName来获得关联funcs
        callee_df = df_merged[(df_merged['type'] == "method") & (df_merged['name'].isin(group["endName"].tolist()))]
        # todo 这里要注意有可能存在相同的name，并且要想办法区分出来
        caller_to_callee[unique_start_EntityID] = callee_df["unique_entityID"].tolist()

    # 对于不存在于relation表中的函数，需要手动将其进行添加
    all_func_entities = df_merged[df_merged["type"] == 'method']["unique_entityID"].tolist()
    for func_entity in all_func_entities:
        if func_entity not in caller_to_callee.keys():
            caller_to_callee[func_entity] = []
    # plot_graphviz(caller_to_callee)

    # 拓扑排序
    sorted_nodes = topological_sort(caller_to_callee)
    sorted_nodes.reverse()

    # 生成结果
    # todo 后续考虑update的情况
    func_docs = {}

    temp_dir_path = Path(docs_dir)
    if not temp_dir_path.exists():
        temp_dir_path.mkdir(parents=True)

    for node in tqdm(sorted_nodes, desc="Function Docs In the process of generating..."):
        func_docs = load_from_json(docs_dir + FUNC_DOC_FILE_NAME)
        if node in func_docs.keys():
            continue
        else:
            dependencies = caller_to_callee[node]
            attributes = json.loads(df_merged[df_merged['unique_entityID'] == node].iloc[0]["attributes"])
            dependency_docs = {k: v for k, v in func_docs.items() if k in dependencies}  # 这里其实需要用函数名！！！
            func_docs[node] = creator(base_prompt, attributes, dependency_docs)
            save_to_json(func_docs, docs_dir + FUNC_DOC_FILE_NAME)
    conf.func_docs = func_docs


if __name__ == '__main__':
    pass
