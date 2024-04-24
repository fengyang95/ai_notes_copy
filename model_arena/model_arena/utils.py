import re


def extract_code_snippets(s: str) -> list[tuple[str, str]]:
    code_snippets = re.findall(r"```(\w+?)\n(.*?)```", s, re.DOTALL)
    return code_snippets
