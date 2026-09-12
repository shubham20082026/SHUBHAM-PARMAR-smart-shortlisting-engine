"""
Skill vocabulary + extraction + fuzzy matching.

Design rationale (say this to judges):
- We don't rely on an LLM to "guess" a score. We extract an explicit skill
  vocabulary from the JD text itself using a curated technical dictionary,
  then verify presence in each resume using normalized + fuzzy string
  matching (handles "Express.js" vs "ExpressJS" vs "Express").
- This is the KEYWORD arm of the hybrid system. It is deliberately
  deterministic and explainable -> every match/miss can be printed.
"""

import re
import difflib
from typing import List, Dict, Set

# A curated dictionary of common tech / full-stack skills.
# Extend this list freely for your actual JD's domain before the demo.
SKILL_VOCAB = [
    # Languages
    "python", "javascript", "typescript", "java", "c++", "c#", "go", "ruby", "php", "sql",
    "html", "css", "html5", "css3",
    # Frontend
    "react", "reactjs", "react.js", "vue", "vuejs", "vue.js", "angular", "angularjs",
    "redux", "next.js", "nextjs", "tailwind", "tailwindcss", "bootstrap", "jquery",
    # Backend
    "node", "node.js", "nodejs", "express", "express.js", "expressjs", "django", "flask",
    "fastapi", "spring", "spring boot", "laravel", "ruby on rails", ".net", "asp.net",
    # Databases
    "mongodb", "mysql", "postgresql", "postgres", "sqlite", "redis", "firebase",
    "dynamodb", "cassandra", "oracle",
    # APIs / architecture
    "rest api", "restful api", "rest", "graphql", "grpc", "microservices", "websocket",
    "api development", "soap",
    # DevOps / cloud
    "docker", "kubernetes", "aws", "azure", "gcp", "google cloud", "ci/cd", "jenkins",
    "github actions", "terraform", "nginx", "linux",
    # Tools / practices
    "git", "github", "gitlab", "agile", "scrum", "jira", "postman", "webpack", "vite",
    "unit testing", "jest", "pytest", "tdd",
    # Data / ML (in case JD touches it)
    "machine learning", "data structures", "algorithms", "oop",
    "object oriented programming", "system design",
]

# Canonical aliasing so variants collapse to one concept for reporting.
CANONICAL = {
    "reactjs": "react", "react.js": "react",
    "vuejs": "vue", "vue.js": "vue", "angularjs": "angular",
    "node.js": "node", "nodejs": "node",
    "express.js": "express", "expressjs": "express",
    "next.js": "nextjs",
    "tailwindcss": "tailwind",
    "postgres": "postgresql",
    "restful api": "rest api", "rest": "rest api",
    "google cloud": "gcp",
    "object oriented programming": "oop",
    "html5": "html", "css3": "css",
}


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+.#/\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonicalize(term: str) -> str:
    t = term.lower().strip()
    return CANONICAL.get(t, t)


def extract_required_skills(jd_text: str) -> List[str]:
    """
    Pull explicit skills mentioned in the JD by scanning for vocabulary hits.
    Returns a de-duplicated, canonicalized list, ordered by first appearance.
    """
    norm = _normalize(jd_text)
    found = []
    seen: Set[str] = set()
    # sort by length desc so multi-word terms ("rest api") match before
    # their substrings ("api") would confuse things
    for skill in sorted(SKILL_VOCAB, key=len, reverse=True):
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, norm):
            canon = canonicalize(skill)
            if canon not in seen:
                seen.add(canon)
                found.append(canon)
    return found


def _fuzzy_present(term: str, haystack_tokens: List[str], threshold: float = 0.86) -> bool:
    """Fuzzy containment check for a (possibly multi-word) term against resume tokens."""
    term_words = term.split()
    joined = " ".join(haystack_tokens)
    if term in joined:
        return True
    # fuzzy match against sliding windows of the same word-length
    n = len(term_words)
    if n == 0:
        return False
    for i in range(len(haystack_tokens) - n + 1):
        window = " ".join(haystack_tokens[i:i + n])
        ratio = difflib.SequenceMatcher(None, term, window).ratio()
        if ratio >= threshold:
            return True
    return False


def match_skills_in_resume(resume_text: str, required_skills: List[str]) -> Dict[str, bool]:
    """
    For each required skill, determine if it's present in the resume
    (normalized exact match OR fuzzy match to catch typos/variants).
    Returns {skill: True/False}.
    """
    norm = _normalize(resume_text)
    tokens = norm.split()
    results = {}
    for skill in required_skills:
        canon = canonicalize(skill)
        present = _fuzzy_present(canon, tokens)
        # also check known aliases of this canonical skill
        if not present:
            aliases = [k for k, v in CANONICAL.items() if v == canon]
            for alias in aliases:
                if _fuzzy_present(alias, tokens):
                    present = True
                    break
        results[canon] = present
    return results


def keyword_score(resume_text: str, required_skills: List[str]) -> Dict:
    """
    Returns a dict with:
      - score: float in [0,1], fraction of required skills matched
      - matched: list of matched skills
      - missing: list of missing skills
    """
    if not required_skills:
        return {"score": 0.0, "matched": [], "missing": []}
    match_map = match_skills_in_resume(resume_text, required_skills)
    matched = [s for s, ok in match_map.items() if ok]
    missing = [s for s, ok in match_map.items() if not ok]
    score = len(matched) / len(required_skills)
    return {"score": score, "matched": matched, "missing": missing}
