"""
Generates human-readable explanations for the top-N ranked candidates.

Deliberately template-based and grounded in the actual score breakdown
(matched/missing skills, semantic evidence) rather than a free-form LLM
paragraph -> every sentence traces back to a concrete number your system
computed. This is what "quality and clarity of top-3 explanations" wants
to see, and it's trivially defensible in a judge Q&A.

If you want more natural prose, you can pass these structured facts to an
LLM as a LAST formatting step ("turn these bullet facts into 2 sentences")
-- that's fine, because the judgment itself (what matched, what's missing)
already comes from your own system, not the LLM.
"""

from typing import List, Dict


def explain_candidate(result: Dict) -> str:
    cid = result["candidate_id"]
    matched_req = result["matched_required"]
    missing_req = result["missing_required"]
    matched_pref = result["matched_preferred"]
    sem_pct = result["semantic_score"]

    lines = [f"**{cid}** — Rank #{result['rank']} (Score: {result['final_score']}/100)"]

    if matched_req:
        lines.append(f"✅ Matched required skills: {', '.join(matched_req)}")
    else:
        lines.append("✅ Matched required skills: none directly detected")

    if missing_req:
        lines.append(f"⚠️ Missing required skills: {', '.join(missing_req)}")
    else:
        lines.append("⚠️ Missing required skills: none — full required-skill coverage")

    if matched_pref:
        lines.append(f"➕ Bonus/preferred skills also present: {', '.join(matched_pref)}")

    lines.append(
        f"🧠 Semantic fit to JD responsibilities: {sem_pct}% "
        f"(based on best-matching evidence in the resume for each JD requirement line)"
    )

    # cite the single strongest semantic piece of evidence as a concrete example
    if result["top_semantic_matches"]:
        jd_line, resume_line, sim = result["top_semantic_matches"][0]
        lines.append(
            f"   Strongest evidence: JD line \"{jd_line.strip()[:90]}\" "
            f"↔ resume: \"{resume_line.strip()[:90]}\" (similarity {round(sim*100,1)}%)"
        )

    return "\n".join(lines)


def explain_top_n(results: List[Dict], n: int = 3) -> List[str]:
    return [explain_candidate(r) for r in results[:n]]
