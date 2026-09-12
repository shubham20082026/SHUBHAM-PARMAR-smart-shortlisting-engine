"""
Combines the keyword arm and semantic arm into one explainable final score.

Weighting design (defend this choice to judges):
- required-skill keyword coverage: 45%
- preferred-skill keyword coverage: 15%
- semantic fit (meaning-based match to JD requirement lines): 40%

Rationale: the problem statement explicitly says specific tools named in a
JD still matter and shouldn't be satisfied by "only loosely related
experience" -> so keyword coverage of REQUIRED skills gets the single
biggest weight. But semantic fit is what lets genuinely relevant experience
that doesn't use the JD's exact words still score well, so it's weighted
almost as heavily. Preferred/nice-to-have skills matter least.

These weights are constants at the top of the file -> change them in one
place and be ready to explain the numbers to judges.
"""

from typing import List, Dict
from . import skills as skills_mod
from . import semantic as semantic_mod
from . import jd_parser
from . import parser as parser_mod
from . import bias_detector

W_REQUIRED = 0.45
W_PREFERRED = 0.15
W_SEMANTIC = 0.40


def score_candidate(jd_text: str, resume_text: str,
                     required_skills: List[str],
                     preferred_skills: List[str],
                     jd_requirement_chunks: List[str]) -> Dict:
    kw_required = skills_mod.keyword_score(resume_text, required_skills)
    kw_preferred = skills_mod.keyword_score(resume_text, preferred_skills)

    resume_chunks = parser_mod.chunk_text(resume_text)
    sem_score, sem_matches = semantic_mod.semantic_score(jd_requirement_chunks, resume_chunks)

    final = (W_REQUIRED * kw_required["score"]
             + W_PREFERRED * kw_preferred["score"]
             + W_SEMANTIC * sem_score)

    return {
        "final_score": round(final * 100, 1),  # scale to /100 for readability
        "required_skill_score": round(kw_required["score"] * 100, 1),
        "preferred_skill_score": round(kw_preferred["score"] * 100, 1),
        "semantic_score": round(sem_score * 100, 1),
        "matched_required": kw_required["matched"],
        "missing_required": kw_required["missing"],
        "matched_preferred": kw_preferred["matched"],
        "missing_preferred": kw_preferred["missing"],
        "top_semantic_matches": sorted(sem_matches, key=lambda x: -x[2])[:3],
    }


def rank_candidates(jd_text: str, resumes: List[dict]) -> List[Dict]:
    """
    resumes: list of {"id":..., "text":...}
    Returns a list of result dicts sorted best-fit first, each including
    the candidate id and full score breakdown.
    """
    sections = jd_parser.split_jd_sections(jd_text)
    required_text = sections["required"] or jd_text
    preferred_text = sections["preferred"]

    required_skills = skills_mod.extract_required_skills(required_text)
    # if the "required" section detection failed, fall back to whole-JD extraction
    if not required_skills:
        required_skills = skills_mod.extract_required_skills(jd_text)
    preferred_skills = skills_mod.extract_required_skills(preferred_text) if preferred_text else []
    # don't double count a skill as both required and preferred
    preferred_skills = [s for s in preferred_skills if s not in required_skills]

    jd_requirement_chunks = jd_parser.get_requirement_chunks(jd_text)

    results = []
    for r in resumes:
        breakdown = score_candidate(
            jd_text, r["text"], required_skills, preferred_skills, jd_requirement_chunks
        )
        breakdown["candidate_id"] = r["id"]
        results.append(breakdown)

    results.sort(key=lambda x: -x["final_score"])
    for rank, res in enumerate(results, start=1):
        res["rank"] = rank
    return results


def analyze_jd_bias(jd_text: str) -> List[Dict]:
    """
    Bonus feature: flags potential bias / overly narrow phrasing in the JD.
    Separated from rank_candidates so it can be called once per JD rather
    than recomputed per candidate.
    """
    sections = jd_parser.split_jd_sections(jd_text)
    required_text = sections["required"] or jd_text
    required_skills = skills_mod.extract_required_skills(required_text)
    if not required_skills:
        required_skills = skills_mod.extract_required_skills(jd_text)
    return bias_detector.detect_bias(jd_text, required_skills)
