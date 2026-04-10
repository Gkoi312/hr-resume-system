# app/services/semantic_chunk_matching.py
"""
Semantic axes over stored vector chunks (job vs candidate).

`job_chunks` / `cand_chunks` keys are ``profile_type`` (same as
``VectorProfileModel.profile_type``), built by ``app.rag.chunk_profiles``:
job: skill, resp | resp_<n>, jd_raw (fallback), role; candidate: skill,
proj_<n>, work_<n>, cand_role.

Rules:
- Any axis missing on either side → axis score 0 (no reweighting).
- Delivery axis: multiple strong matches should score higher than a single max pair only.
- HR 视角默认更重视项目/实习 ↔ 岗位职责：总分校验权重默认提高 delivery；经历轴单独用 ``cosine_to_experience_score`` 拉开常见余弦区间。
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.rag.hybrid_retrieval import (
    competition_ranks,
    compute_rrf_matrix,
    tokenize,
)
from app.rag.vector_store import _cosine

# Fusion into overall semantic_score (sum normalized at runtime; env can override).
# `role` is kept for diagnostics, but default contribution is disabled because the
# current job/candidate role texts are not yet well-aligned semantically.
_DEFAULT_WEIGHT_SKILL = 0.20
_DEFAULT_WEIGHT_DELIVERY = 0.80
_DEFAULT_WEIGHT_ROLE = 0.00

# Delivery = blend job coverage (每条职责能否被经历对上) with cand breadth (多段经历各自对齐 JD)
_DEFAULT_DELIVERY_JOB_COV = 0.62
_DEFAULT_DELIVERY_CAND_BREADTH = 0.38
DELIVERY_TOPK_CAND_CHUNKS = 3

_SNIP = 320


def _delivery_hybrid_enabled() -> bool:
    v = (os.getenv("SEMANTIC_DELIVERY_HYBRID") or "1").strip().lower()
    return v not in ("0", "false", "no", "off", "n")


def _rrf_k() -> int:
    try:
        return max(1, int(os.getenv("SEMANTIC_RRF_K") or "60"))
    except ValueError:
        return 60


def _semantic_fusion_weights() -> Tuple[float, float, float]:
    """skill, delivery, role — normalized to sum 1.0. Env: SEMANTIC_WEIGHT_*."""
    def _f(name: str, default: float) -> float:
        raw = (os.getenv(name) or "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    w_s = _f("SEMANTIC_WEIGHT_SKILL", _DEFAULT_WEIGHT_SKILL)
    w_d = _f("SEMANTIC_WEIGHT_DELIVERY", _DEFAULT_WEIGHT_DELIVERY)
    w_r = _f("SEMANTIC_WEIGHT_ROLE", _DEFAULT_WEIGHT_ROLE)
    s = w_s + w_d + w_r
    if s <= 0:
        return (
            _DEFAULT_WEIGHT_SKILL,
            _DEFAULT_WEIGHT_DELIVERY,
            _DEFAULT_WEIGHT_ROLE,
        )
    return w_s / s, w_d / s, w_r / s


def _delivery_coverage_weights() -> Tuple[float, float]:
    """Job-coverage vs candidate-breadth within delivery axis. Env: SEMANTIC_DELIVERY_JOB_COVERAGE (0–1)."""
    raw = (os.getenv("SEMANTIC_DELIVERY_JOB_COVERAGE") or "").strip()
    if not raw:
        return _DEFAULT_DELIVERY_JOB_COV, _DEFAULT_DELIVERY_CAND_BREADTH
    try:
        wj = float(raw)
        wj = max(0.0, min(1.0, wj))
        return wj, 1.0 - wj
    except ValueError:
        return _DEFAULT_DELIVERY_JOB_COV, _DEFAULT_DELIVERY_CAND_BREADTH


def cosine_to_band_score(sim: float) -> float:
    if sim <= 0:
        return 0.0
    if sim >= 0.84:
        return 95.0
    if sim >= 0.78:
        return 85.0
    if sim >= 0.72:
        return 75.0
    if sim >= 0.66:
        return 65.0
    return 55.0


def cosine_to_experience_score(sim: float) -> float:
    """
    Project/work ↔ JD 职责专用：在 HR 常见的余弦区间 (约 0.4–0.66) 用线性拉伸，
    避免全部落在 band=55；高段仍与技能轴高档对齐。
    """
    if sim <= 0:
        return 0.0
    if sim >= 0.84:
        return 95.0
    if sim >= 0.78:
        return 88.0
    if sim >= 0.72:
        return 80.0
    if sim >= 0.66:
        return 72.0
    lo, hi = 0.38, 0.66
    lo_score, hi_score = 38.0, 72.0
    t = (sim - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    return round(lo_score + t * (hi_score - lo_score), 1)


def _vec_nonempty(vec: List[float]) -> bool:
    return bool(vec)


def _job_delivery_chunks(
    job_chunks: Dict[str, Tuple[List[float], Dict[str, Any]]],
) -> List[Tuple[str, List[float], Dict[str, Any]]]:
    items: List[Tuple[str, List[float], Dict[str, Any]]] = []
    for profile_type, (vec, meta) in job_chunks.items():
        # build_job_chunks: responsibilities → "resp" or "resp_<i>"
        if profile_type == "resp" or profile_type.startswith("resp_"):
            if _vec_nonempty(vec):
                items.append((profile_type, vec, meta))
    if items:
        return items
    jd = job_chunks.get("jd_raw")
    if jd and _vec_nonempty(jd[0]):
        return [("jd_raw", jd[0], jd[1])]
    return []


def _cand_experience_chunks(
    cand_chunks: Dict[str, Tuple[List[float], Dict[str, Any]]],
) -> List[Tuple[str, List[float], Dict[str, Any]]]:
    out: List[Tuple[str, List[float], Dict[str, Any]]] = []
    for profile_type, (vec, meta) in cand_chunks.items():
        # build_candidate_chunks: project → proj_<i>; work_experience (全职/实习同一列表) → work_<i>
        if profile_type.startswith("proj_") or profile_type.startswith("work_"):
            if _vec_nonempty(vec):
                out.append((profile_type, vec, meta))
    return out


def axis_skill_score(
    job_chunks: Dict[str, Tuple[List[float], Dict[str, Any]]],
    cand_chunks: Dict[str, Tuple[List[float], Dict[str, Any]]],
) -> Tuple[float, Optional[float], bool]:
    """Returns (0-100 band, raw cosine or None, axis_complete)."""
    job_skill = job_chunks.get("skill")
    cand_skill = cand_chunks.get("skill")
    # Each value is (vector, meta) from get_entity_chunks / get_candidate_chunks_bulk.
    if (
        not job_skill
        or not cand_skill
        or not _vec_nonempty(job_skill[0])
        or not _vec_nonempty(cand_skill[0])
    ):
        return 0.0, None, False
    sim = _cosine(job_skill[0], cand_skill[0])
    return cosine_to_band_score(sim), sim, True


def axis_role_score(
    job_chunks: Dict[str, Tuple[List[float], Dict[str, Any]]],
    cand_chunks: Dict[str, Tuple[List[float], Dict[str, Any]]],
) -> Tuple[float, Optional[float], bool]:
    job_role = job_chunks.get("role")
    cand_role = cand_chunks.get("cand_role")
    if (
        not job_role
        or not cand_role
        or not _vec_nonempty(job_role[0])
        or not _vec_nonempty(cand_role[0])
    ):
        return 0.0, None, False
    sim = _cosine(job_role[0], cand_role[0])
    return cosine_to_band_score(sim), sim, True


def axis_delivery_score(
    job_chunks: Dict[str, Tuple[List[float], Dict[str, Any]]],
    cand_chunks: Dict[str, Tuple[List[float], Dict[str, Any]]],
) -> Tuple[float, Dict[str, Any]]:
    """
    Delivery / 经历轴：
    - Job 侧：每条职责（``resp`` / ``resp_<n>``；无则 ``jd_raw``）与所有 ``proj_*`` / ``work_*``
      对齐：默认用 **BM25 + 余弦 RRF** 选对 (职责→最佳经历)，再用该对的 **余弦** 映射 band，
      各职责 band 平均。
    - Candidate 侧：每段经历对全部职责用 RRF 取 max，再对经历段取 top-K 的 band 平均。

    关闭混合检索：``SEMANTIC_DELIVERY_HYBRID=0`` → 退回纯 max 余弦。
    """
    job_items = _job_delivery_chunks(job_chunks)
    cand_items = _cand_experience_chunks(cand_chunks)
    detail: Dict[str, Any] = {
        "job_delivery_profile_types": [pt for pt, _, _ in job_items],
        "cand_experience_profile_types": [pt for pt, _, _ in cand_items],
    }
    if not job_items or not cand_items:
        detail["axis_empty"] = True
        return 0.0, detail

    j_count = len(job_items)
    d_count = len(cand_items)
    cos_matrix: List[List[float]] = [
        [_cosine(job_items[j][1], cand_items[d][1]) for d in range(d_count)]
        for j in range(j_count)
    ]

    if not _delivery_hybrid_enabled():
        return _axis_delivery_pure_cosine(
            job_items, cand_items, cos_matrix, detail
        )

    job_texts = [
        str((job_items[j][2] or {}).get("semantic_text", "") or "") for j in range(j_count)
    ]
    cand_texts = [
        str((cand_items[d][2] or {}).get("semantic_text", "") or "") for d in range(d_count)
    ]
    job_qtok = [tokenize(t) for t in job_texts]
    cand_tok = [tokenize(t) for t in cand_texts]

    rrf_k = _rrf_k()
    rrf_matrix, bm25_degenerate_rows, bm25_rows = compute_rrf_matrix(
        cos_matrix,
        job_qtok,
        cand_tok,
        rrf_k=rrf_k,
    )

    per_job_sims: List[float] = []
    delivery_alignments: List[Dict[str, Any]] = []
    for j in range(j_count):
        row_rrf = rrf_matrix[j]
        d_best = max(
            range(d_count),
            key=lambda d: (row_rrf[d], cos_matrix[j][d], -d),
        )
        per_job_sims.append(cos_matrix[j][d_best])
        row_cos = cos_matrix[j]
        bm25_row = bm25_rows[j]
        rank_cos = competition_ranks(row_cos, higher_is_better=True)
        rank_bm25 = competition_ranks(bm25_row, higher_is_better=True)
        jpt, _, _ = job_items[j]
        cpt, _, _ = cand_items[d_best]
        jtxt = job_texts[j]
        ctxt = cand_texts[d_best]
        delivery_alignments.append(
            {
                "job_profile_type": jpt,
                "cand_profile_type": cpt,
                "cosine": round(row_cos[d_best], 6),
                "bm25": round(bm25_row[d_best], 6),
                "rank_cos": rank_cos[d_best],
                "rank_bm25": rank_bm25[d_best],
                "rrf": round(row_rrf[d_best], 8),
                "bm25_degenerate": bm25_degenerate_rows[j],
                "job_text_snippet": jtxt[:_SNIP] + ("…" if len(jtxt) > _SNIP else ""),
                "cand_text_snippet": ctxt[:_SNIP] + ("…" if len(ctxt) > _SNIP else ""),
            }
        )

    per_job_xp = [cosine_to_experience_score(s) for s in per_job_sims]
    mean_job_xp = sum(per_job_xp) / len(per_job_xp)

    per_cand_best_rrf: List[float] = []
    per_cand_best_cos: List[float] = []
    for d in range(d_count):
        best_j = max(
            range(j_count),
            key=lambda j: (rrf_matrix[j][d], cos_matrix[j][d], -j),
        )
        per_cand_best_rrf.append(rrf_matrix[best_j][d])
        per_cand_best_cos.append(cos_matrix[best_j][d])

    order = sorted(
        range(d_count),
        key=lambda d: (per_cand_best_rrf[d], per_cand_best_cos[d]),
        reverse=True,
    )
    k = min(DELIVERY_TOPK_CAND_CHUNKS, d_count)
    top_idx = order[:k]
    mean_cand_xp = (
        sum(cosine_to_experience_score(per_cand_best_cos[d]) for d in top_idx) / k
    )

    w_job, w_cand = _delivery_coverage_weights()
    delivery = round(w_job * mean_job_xp + w_cand * mean_cand_xp, 1)
    detail.update(
        {
            "axis_empty": False,
            "delivery_fusion": "rrf",
            "rrf_k": rrf_k,
            "per_job_max_cosine": per_job_sims,
            "per_cand_max_cosine": sorted(per_cand_best_cos, reverse=True),
            "per_cand_max_rrf": [round(per_cand_best_rrf[d], 8) for d in order],
            "mean_job_cosine": round(sum(per_job_sims) / len(per_job_sims), 4),
            "mean_cand_topk_cosine": round(
                sum(per_cand_best_cos[d] for d in top_idx) / k, 4
            ),
            "mean_job_experience_score": round(mean_job_xp, 1),
            "mean_cand_topk_experience_score": round(mean_cand_xp, 1),
            "mean_job_band": round(mean_job_xp, 1),
            "mean_cand_topk_band": round(mean_cand_xp, 1),
            "delivery_topk": k,
            "delivery_alignments": delivery_alignments,
        }
    )
    return delivery, detail


def _axis_delivery_pure_cosine(
    job_items: List[Tuple[str, List[float], Dict[str, Any]]],
    cand_items: List[Tuple[str, List[float], Dict[str, Any]]],
    cos_matrix: List[List[float]],
    detail: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    j_count = len(job_items)
    d_count = len(cand_items)
    per_job_sims = [max(cos_matrix[j][d] for d in range(d_count)) for j in range(j_count)]
    per_job_xp = [cosine_to_experience_score(s) for s in per_job_sims]
    mean_job_xp = sum(per_job_xp) / len(per_job_xp)

    per_cand_sims = [max(cos_matrix[j][d] for j in range(j_count)) for d in range(d_count)]
    per_cand_sims_sorted = sorted(per_cand_sims, reverse=True)
    k = min(DELIVERY_TOPK_CAND_CHUNKS, len(per_cand_sims_sorted))
    topk = per_cand_sims_sorted[:k]
    mean_cand_xp = sum(cosine_to_experience_score(s) for s in topk) / k

    w_job, w_cand = _delivery_coverage_weights()
    delivery = round(w_job * mean_job_xp + w_cand * mean_cand_xp, 1)
    detail.update(
        {
            "axis_empty": False,
            "delivery_fusion": "cosine_only",
            "per_job_max_cosine": per_job_sims,
            "per_cand_max_cosine": per_cand_sims_sorted,
            "mean_job_cosine": round(sum(per_job_sims) / len(per_job_sims), 4),
            "mean_cand_topk_cosine": round(sum(topk) / k, 4),
            "mean_job_experience_score": round(mean_job_xp, 1),
            "mean_cand_topk_experience_score": round(mean_cand_xp, 1),
            "mean_job_band": round(mean_job_xp, 1),
            "mean_cand_topk_band": round(mean_cand_xp, 1),
            "delivery_topk": k,
        }
    )
    return delivery, detail


def compute_semantic_scores_for_candidate_chunks(
    job_chunks: Dict[str, Tuple[List[float], Dict[str, Any]]],
    cand_chunks: Dict[str, Tuple[List[float], Dict[str, Any]]],
) -> Tuple[float, Dict[str, Any]]:
    """One candidate: overall semantic 0-100 + diagnostics for API / explanation."""
    skill_band, skill_sim, skill_ok = axis_skill_score(job_chunks, cand_chunks)
    role_band, role_sim, role_ok = axis_role_score(job_chunks, cand_chunks)
    delivery_band, delivery_detail = axis_delivery_score(job_chunks, cand_chunks)
    delivery_ok = not delivery_detail.get("axis_empty")

    w_skill, w_delivery, w_role = _semantic_fusion_weights()
    semantic_score = round(
        w_skill * skill_band
        + w_delivery * delivery_band
        + w_role * role_band,
        1,
    )

    semantic_status = "available" if any(cand_chunks.keys()) else "not_indexed"

    out_detail: Dict[str, Any] = {
        "semantic_status": semantic_status,
        "skill_similarity": skill_sim,
        "skill_band": skill_band,
        "skill_axis_complete": skill_ok,
        "role_similarity": role_sim,
        "role_band": role_band,
        "role_axis_complete": role_ok,
        "delivery_band": delivery_band,
        "delivery_axis_complete": delivery_ok,
        "semantic_weight_skill": w_skill,
        "semantic_weight_delivery": w_delivery,
        "semantic_weight_role": w_role,
        "delivery_detail": delivery_detail,
        "evidence_snippets": _build_evidence_snippets(
            job_chunks, cand_chunks, skill_sim, role_sim, delivery_detail
        ),
    }
    return semantic_score, out_detail


def _build_evidence_snippets(
    job_chunks: Dict[str, Tuple[List[float], Dict[str, Any]]],
    cand_chunks: Dict[str, Tuple[List[float], Dict[str, Any]]],
    skill_sim: Optional[float],
    role_sim: Optional[float],
    delivery_detail: Dict[str, Any],
) -> List[Dict[str, Any]]:
    snippets: List[Dict[str, Any]] = []
    cs = cand_chunks.get("skill")
    if skill_sim is not None and cs:
        t = (cs[1] or {}).get("semantic_text", "")
        if t:
            snippets.append({"source_type": "skill", "text": t, "similarity": skill_sim})
    cr = cand_chunks.get("cand_role")
    if role_sim is not None and cr:
        t = (cr[1] or {}).get("semantic_text", "")
        if t:
            snippets.append({"source_type": "role", "text": t, "similarity": role_sim})

    job_items = _job_delivery_chunks(job_chunks)
    cand_items = _cand_experience_chunks(cand_chunks)
    if job_items and cand_items and not delivery_detail.get("axis_empty"):
        alignments = delivery_detail.get("delivery_alignments")
        if isinstance(alignments, list) and alignments:
            best = max(
                (a for a in alignments if isinstance(a, dict)),
                key=lambda a: (float(a.get("rrf") or 0.0), float(a.get("cosine") or 0.0)),
            )
            ct = str(best.get("cand_text_snippet") or "")
            if ct:
                jpt = str(best.get("job_profile_type") or "")
                cpt = str(best.get("cand_profile_type") or "")
                rrf_v = best.get("rrf")
                deg = best.get("bm25_degenerate")
                snippets.append(
                    {
                        "source_type": "delivery",
                        "text": ct,
                        "similarity": float(best.get("cosine") or 0.0),
                        "rationale": (
                            f"职责块 {jpt} ↔ 经历块 {cpt}；RRF={rrf_v}；"
                            f"词法{'退化仅用向量排名' if deg else 'BM25+向量融合'}"
                        ),
                    }
                )
        else:
            best_sim = -1.0
            best_text = ""
            for _pj, jvec, _ in job_items:
                for _pc, cvec, cmeta in cand_items:
                    sim = _cosine(jvec, cvec)
                    if sim > best_sim:
                        best_sim = sim
                        best_text = cmeta.get("semantic_text", "")
            if best_sim >= 0 and best_text:
                snippets.append(
                    {
                        "source_type": "delivery",
                        "text": best_text,
                        "similarity": best_sim,
                    }
                )
    return snippets


async def compute_semantic_scores_for_candidates_bulk(
    job_id: uuid.UUID,
    candidates: List[Any],
    vector_store: Any,
) -> Dict[uuid.UUID, Tuple[float, Dict[str, Any]]]:
    """Load job chunks once; bulk candidate chunks; map candidate_id → (score, detail)."""
    if not candidates:
        return {}
    job_chunks = await vector_store.get_entity_chunks("job", job_id, status="available")
    ids = [c.id for c in candidates]
    bulk = await vector_store.get_candidate_chunks_bulk(ids, status="available")
    out: Dict[uuid.UUID, Tuple[float, Dict[str, Any]]] = {}
    for c in candidates:
        cid = c.id
        cand_chunks = bulk.get(cid, {})
        score, detail = compute_semantic_scores_for_candidate_chunks(job_chunks, cand_chunks)
        out[cid] = (score, detail)
    return out
