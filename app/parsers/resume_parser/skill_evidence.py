"""Skill normalization + lexicon extraction from resume text (no evidence gating)."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

# Canonical skill lexicon — ONLY true synonyms (same thing, different name).
# Frameworks, libraries, and tools are intentionally NOT merged into their parent
# language/platform: "django" stays "django", not collapsed into "python".
#
# Rule: if two terms refer to the *exact same thing*, they share a canonical.
#   ✅ k8s = kubernetes          (same product, abbreviation)
#   ✅ golang = go               (same language, alias)
#   ❌ django → python           (framework ≠ language — different things)
#   ❌ postgresql → sql          (specific db ≠ general concept)
_SKILL_LEXICON: List[Tuple[str, Tuple[str, ...]]] = [
    # ---- languages: only abbreviation / alias variants ----
    ("python", ("python",)),
    ("java", ("java",)),
    ("go", ("go", "golang")),
    ("c++", ("c++", "cpp")),
    ("rust", ("rust",)),
    ("javascript", ("javascript", "js")),
    ("typescript", ("typescript", "ts")),
    ("kotlin", ("kotlin",)),
    ("swift", ("swift",)),
    ("objective-c", ("objective-c", "objc", "objective c")),
    # ---- databases & data stores ----
    ("sql", ("sql",)),
    ("mysql", ("mysql", "mariadb")),
    ("postgresql", ("postgresql", "postgres")),
    ("oracle", ("oracle", "oracle database")),
    ("sqlite", ("sqlite",)),
    ("redis", ("redis",)),
    ("mongodb", ("mongodb", "mongo")),
    ("cassandra", ("cassandra",)),
    ("elasticsearch", ("elasticsearch", "es", "elastic")),
    # ---- message queues & streaming ----
    ("kafka", ("kafka",)),
    ("rabbitmq", ("rabbitmq",)),
    ("activemq", ("activemq",)),
    # ---- containers & orchestration ----
    ("docker", ("docker",)),
    ("kubernetes", ("kubernetes", "k8s")),
    # ---- CI/CD & version control ----
    ("git", ("git",)),
    ("github", ("github",)),
    ("gitlab", ("gitlab",)),
    ("jenkins", ("jenkins",)),
    ("github actions", ("github actions", "gh actions")),
    # ---- OS & scripting ----
    ("linux", ("linux",)),
    ("bash", ("bash", "shell script", "shell scripting")),
    # ---- ML / AI frameworks ----
    ("pytorch", ("pytorch", "torch")),
    ("tensorflow", ("tensorflow", "tf")),
    ("scikit-learn", ("scikit-learn", "scikit learn", "sklearn")),
    ("xgboost", ("xgboost", "xgb")),
    ("lightgbm", ("lightgbm", "lgb")),
    # ---- ML domains (Chinese ↔ English) ----
    ("机器学习", ("机器学习", "machine learning", "ml")),
    ("深度学习", ("深度学习", "deep learning", "dl")),
    ("nlp", ("nlp", "natural language processing", "自然语言处理", "自然语言")),
    ("cv", ("cv", "computer vision", "计算机视觉")),
    ("大模型", ("大模型", "llm", "large language model")),
    # ---- big data ----
    ("spark", ("spark", "apache spark")),
    ("hadoop", ("hadoop",)),
    ("hive", ("hive",)),
    ("flink", ("flink",)),
    # ---- data analysis ----
    ("数据分析", ("数据分析", "data analysis")),
    ("tableau", ("tableau",)),
    ("power bi", ("power bi", "powerbi")),
    ("excel", ("excel",)),
    # ---- mobile ----
    ("android", ("android",)),
    ("ios", ("ios",)),
    # ---- web frameworks (keep distinct) ----
    ("django", ("django",)),
    ("flask", ("flask",)),
    ("fastapi", ("fastapi",)),
    ("spring", ("spring", "spring framework")),
    ("spring boot", ("spring boot", "springboot")),
    ("vue", ("vue", "vue.js", "vuejs")),
    ("react", ("react", "react.js", "reactjs")),
    ("angular", ("angular", "angular.js", "angularjs")),
    ("node.js", ("node.js", "nodejs", "node")),
    # ---- web servers / reverse proxies ----
    ("nginx", ("nginx",)),
    ("apache", ("apache", "apache httpd", "httpd")),
    # ---- cloud platforms ----
    ("aws", ("aws", "amazon web services")),
    ("azure", ("azure", "microsoft azure")),
    ("gcp", ("gcp", "google cloud", "google cloud platform")),
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
