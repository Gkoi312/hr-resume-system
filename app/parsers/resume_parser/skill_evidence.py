"""Skill normalization + lexicon extraction from resume text (no evidence gating)."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

_SKILL_LEXICON: List[Tuple[str, Tuple[str, ...]]] = [
    ("python", ("python", "django", "flask", "fastapi", "pandas", "numpy", "sklearn")),
    ("java", ("java", "spring", "springboot", "spring boot", "mybatis", "maven", "gradle")),
    ("go", ("golang", "gin框架", " gin ", " go语言", "go语言")),
    ("c++", ("c++", "cpp", "cmake")),
    ("c", (" c语言", "c语言", " embedded c")),
    ("rust", ("rust", "tokio", "cargo")),
    ("javascript", ("javascript", "typescript", "nodejs", "node.js", "vue", "react", "angular")),
    ("sql", ("sql", "mysql", "postgresql", "postgres", "oracle", "sqlite", "数据库")),
    ("mysql", ("mysql", " mariadb")),
    ("redis", ("redis",)),
    ("mongodb", ("mongodb", "mongo")),
    ("kafka", ("kafka",)),
    ("rabbitmq", ("rabbitmq", "amqp")),
    ("docker", ("docker", "kubernetes", "k8s", "容器")),
    ("linux", ("linux", "ubuntu", "centos", "shell", "bash")),
    ("git", ("git", "github", "gitlab", "ci/cd", "jenkins")),
    ("pytorch", ("pytorch", "torch")),
    ("tensorflow", ("tensorflow", "keras")),
    ("机器学习", ("机器学习", "machine learning", "ml", "xgboost", "lightgbm")),
    ("深度学习", ("深度学习", "deep learning", "神经网络", "cnn", "rnn", "transformer")),
    ("nlp", ("nlp", "自然语言", "bert", "llm", "大模型")),
    ("cv", ("计算机视觉", "opencv", "图像识别", "目标检测")),
    ("spark", ("spark", "hadoop", "hive", "flink")),
    ("数据分析", ("数据分析", "tableau", "power bi", "excel")),
    ("android", ("android", "kotlin")),
    ("ios", ("ios", "swift", "objective-c", "objc")),
]


def _nfkc_lower(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").lower()


def _normalize_skill_token(raw: str) -> str:
    t = unicodedata.normalize("NFKC", (raw or "").strip())
    if not t:
        return ""
    if re.search(r"[\u4e00-\u9fff]", t):
        return t[:24] if len(t) > 24 else t
    return t.lower()


def _lexicon_index() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for canon, aliases in _SKILL_LEXICON:
        for a in aliases:
            key = _nfkc_lower(a.strip())
            if key:
                out[key] = canon
        ck = _nfkc_lower(canon)
        if ck:
            out[ck] = canon
    return out


_ALIAS_TO_CANONICAL = _lexicon_index()


def map_phrase_to_canonical(phrase: str) -> Optional[str]:
    pl = _nfkc_lower(phrase.strip())
    if not pl:
        return None
    if pl in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[pl]
    for alias, canon in _ALIAS_TO_CANONICAL.items():
        if len(alias) >= 2 and alias in pl:
            return canon
    return None


def normalize_skill_list(raw: List[str]) -> List[str]:
    """Dedupe resume header skills; map aliases to lexicon canonicals where possible."""
    seen: Set[str] = set()
    out: List[str] = []
    for x in raw:
        t = _normalize_skill_token(str(x))
        if not t:
            continue
        c = map_phrase_to_canonical(t)
        fin = c if c else (t.lower() if re.match(r"^[a-z0-9.+#\\-]+$", t) else t)
        if fin not in seen:
            seen.add(fin)
            out.append(fin)
    return out
