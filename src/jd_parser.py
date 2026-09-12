"""
Splits a JD into individual "requirement chunks" (one per responsibility /
required skill / qualification line) so semantic matching can be done at
a granular, per-requirement level instead of one big blob.

Also separates "required/must-have" lines from "nice-to-have/preferred"
lines using common header keywords, so keyword scoring can weight them
differently (a role that explicitly requires a skill shouldn't be
satisfied by only loosely related experience - per the problem statement).
"""

import re
from typing import List, Dict

REQUIRED_HEADERS = [
    "requirements", "required skills", "must have", "must-have",
    "qualifications", "what you need", "you should have", "required",
]
PREFERRED_HEADERS = [
    "nice to have", "nice-to-have", "preferred", "bonus", "good to have",
    "plus", "optional",
]


def split_jd_sections(jd_text: str) -> Dict[str, str]:
    """
    Naive section splitter: scans line by line, switches "current section"
    when it sees a header keyword, and buckets subsequent lines under
    'required', 'preferred', or 'general' (responsibilities/description).
    """
    lines = [l.strip() for l in jd_text.split("\n") if l.strip()]
    sections = {"required": [], "preferred": [], "general": []}
    current = "general"
    for line in lines:
        low = line.lower()
        if any(h in low for h in REQUIRED_HEADERS) and len(line) < 60:
            current = "required"
            continue
        if any(h in low for h in PREFERRED_HEADERS) and len(line) < 60:
            current = "preferred"
            continue
        sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


def get_requirement_chunks(jd_text: str) -> List[str]:
    """
    Returns a list of bullet/sentence-level requirement chunks across the
    whole JD (used for the semantic matching arm - we want per-requirement
    granularity, not one giant JD embedding).
    """
    # split on bullets, newlines, and sentence boundaries
    raw = re.split(r"\n+|•|\u2022|- (?=[A-Za-z])|(?<=[.;])\s+", jd_text)
    chunks = []
    for r in raw:
        r = r.strip(" \t-•")
        if len(r.split()) >= 3:
            chunks.append(r)
    return chunks if chunks else [jd_text]
