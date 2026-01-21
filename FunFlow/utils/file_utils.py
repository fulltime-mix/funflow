"""File I/O utilities"""

import json


def read_jsonl(jsonl_file):
    """Read JSONL file and return list of parsed objects"""
    lists = []
    with open(jsonl_file, "r", encoding="utf8") as fin:
        for line in fin:
            if line.strip():
                lists.append(json.loads(line.strip()))
    return lists
