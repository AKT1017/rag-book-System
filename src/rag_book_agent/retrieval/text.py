import re
from typing import List


def tokens(text: str) -> List[str]:
    text = text.lower()
    words = re.findall(r"[a-z0-9_]+", text)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    chinese = []
    for run in chinese_runs:
        chinese.extend(list(run))
        chinese.extend(run[index : index + 2] for index in range(len(run) - 1))
    return words + chinese


def fts_query(text: str) -> str:
    words = re.findall(r"[a-zA-Z0-9_]+", text)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    terms = words + chinese_runs
    safe_terms = ['"%s"' % term.replace('"', "") for term in terms if term]
    return " OR ".join(safe_terms)
