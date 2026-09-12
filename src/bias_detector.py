"""
Flags potential bias or overly narrow phrasing in a JD that could unfairly
exclude otherwise qualified candidates.

This is rule-based and explainable (same philosophy as the rest of the
system) -> every flag traces back to a concrete phrase in the JD and a
named reason, so it's defensible in front of judges.

Categories checked:
1. Gendered / exclusionary language (masculine-coded words known from HR
   research to discourage women from applying, e.g. "ninja", "rockstar",
   "aggressive", "dominant").
2. Over-specific tool requirements with no "or equivalent" language, where
   a more general skill would capture the same competency (e.g. demanding
   "React" by name for a role that's really about "modern frontend
   framework experience").
3. Unreasonable experience/education floors for the seniority implied by
   the title (e.g. "5+ years" or "Bachelor's required" on an intern/junior
   posting).
4. Excessive simultaneous "must-have" tool lists (a wall of 8+ required
   named technologies) that filters out otherwise strong generalists.
"""

import re
from typing import List, Dict

GENDER_CODED_MASCULINE = [
    "ninja", "rockstar", "guru", "hacker", "dominant", "aggressive",
    "competitive", "fearless", "superhero",
]

GENERAL_VS_SPECIFIC_TOOLS = {
    "react": "modern frontend framework (React/Vue/Angular)",
    "node": "backend JavaScript runtime",
    "node.js": "backend JavaScript runtime",
    "express": "Node.js web framework (Express/Koa/Fastify)",
    "mongodb": "NoSQL database (MongoDB/DynamoDB/Firebase)",
    "aws": "cloud platform (AWS/Azure/GCP)",
    "docker": "containerization tool (Docker/Podman)",
}

JUNIOR_TITLE_HINTS = ["intern", "junior", "entry level", "entry-level", "graduate", "trainee"]

EXPERIENCE_PATTERN = re.compile(r"(\d+)\s*\+?\s*(?:years|yrs)", re.IGNORECASE)


def detect_bias(jd_text: str, required_skills: List[str]) -> List[Dict]:
    flags = []
    lower = jd_text.lower()
    is_junior_role = any(h in lower for h in JUNIOR_TITLE_HINTS)

    # 1. Gendered / exclusionary language
    for word in GENDER_CODED_MASCULINE:
        if re.search(r"\b" + re.escape(word) + r"\b", lower):
            flags.append({
                "type": "exclusionary_language",
                "detail": f'The word "{word}" is known to discourage some qualified '
                          f'candidates (particularly women) from applying, per HR research '
                          f'on gendered job-ad language. Consider a neutral alternative.',
            })

    # 2. Over-specific tool requirements with no "or equivalent" hedge
    for skill in required_skills:
        general = GENERAL_VS_SPECIFIC_TOOLS.get(skill)
        if general and "or equivalent" not in lower and "similar" not in lower:
            flags.append({
                "type": "overly_narrow_tool_requirement",
                "detail": f'Requiring "{skill}" by name (with no "or equivalent" wording) '
                          f'may exclude candidates with directly transferable skills in a '
                          f'{general}. Consider phrasing it as the general competency with '
                          f'"{skill}" as an example.',
            })

    # 3. Experience/education floor mismatched to seniority
    if is_junior_role:
        for match in EXPERIENCE_PATTERN.finditer(jd_text):
            years = int(match.group(1))
            if years >= 2:
                flags.append({
                    "type": "seniority_mismatch",
                    "detail": f'The JD asks for {years}+ years of experience but the title '
                              f'suggests an intern/junior/entry-level role. This floor may '
                              f'exclude the very candidates the role is meant for.',
                })
        if re.search(r"bachelor'?s?\s+(?:degree\s+)?required", lower):
            flags.append({
                "type": "seniority_mismatch",
                "detail": 'A required (non-negotiable) Bachelor\'s degree on an intern/junior '
                          'posting can exclude strong self-taught or bootcamp candidates. '
                          'Consider "preferred" instead of "required" if the role is truly entry-level.',
            })

    # 4. Excessive simultaneous required tools
    if len(required_skills) >= 8:
        flags.append({
            "type": "excessive_requirements",
            "detail": f'{len(required_skills)} named required tools/skills is a long list for '
                      f'one role. Long "must-have" lists tend to filter out strong generalists '
                      f'who could ramp up quickly. Consider splitting into "required" (2-4 core '
                      f'skills) vs "nice to have" (the rest).',
        })

    return flags


def format_bias_report(flags: List[Dict]) -> str:
    if not flags:
        return "No obvious bias or overly narrow phrasing detected in this JD."
    lines = [f"⚠️ {len(flags)} potential issue(s) flagged in the JD:\n"]
    for f in flags:
        lines.append(f"- **[{f['type']}]** {f['detail']}")
    return "\n".join(lines)
