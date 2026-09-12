"""
Streamlit demo UI.

Run with:  streamlit run app.py

Lets you upload a JD PDF + a batch of resume PDFs, see the ranked
shortlist with score breakdowns, read top-3 explanations, and (bonus
feature) ask a natural-language question like "Why is X ranked above Y?"
"""

import os
import re
import tempfile

import streamlit as st

from src import parser as parser_mod
from src import scorer
from src import explainer
from src import bias_detector

st.set_page_config(page_title="Smart Shortlisting Engine", layout="wide")
st.title("📋 Smart Shortlisting Engine")
st.caption("Hybrid semantic + keyword resume ranking against a job description")

with st.sidebar:
    st.header("1. Upload inputs")
    jd_file = st.file_uploader("Job Description (PDF)", type=["pdf"])
    resume_files = st.file_uploader("Resumes (PDF, multiple)", type=["pdf"], accept_multiple_files=True)
    run = st.button("Run ranking", type="primary", disabled=not (jd_file and resume_files))

if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.jd_text = None

if run:
    with tempfile.TemporaryDirectory() as tmp:
        jd_path = os.path.join(tmp, "jd.pdf")
        with open(jd_path, "wb") as f:
            f.write(jd_file.read())
        jd_text = parser_mod.extract_text_from_pdf(jd_path)

        resumes = []
        for rf in resume_files:
            rpath = os.path.join(tmp, rf.name)
            with open(rpath, "wb") as f:
                f.write(rf.read())
            text = parser_mod.extract_text_from_pdf(rpath)
            resumes.append({"id": os.path.splitext(rf.name)[0], "text": text})

        with st.spinner("Extracting skills, embedding text, and scoring candidates..."):
            results = scorer.rank_candidates(jd_text, resumes)

        st.session_state.results = results
        st.session_state.jd_text = jd_text
        st.session_state.resumes_by_id = {r["id"]: r["text"] for r in resumes}
        st.session_state.bias_flags = scorer.analyze_jd_bias(jd_text)

if st.session_state.results:
    results = st.session_state.results

    st.header("Ranked Shortlist")
    table_rows = [{
        "Rank": r["rank"], "Candidate": r["candidate_id"], "Final Score": r["final_score"],
        "Required-skill %": r["required_skill_score"], "Preferred-skill %": r["preferred_skill_score"],
        "Semantic %": r["semantic_score"],
    } for r in results]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    st.bar_chart({r["candidate_id"]: r["final_score"] for r in results})

    st.header("Top 3 — Explanations")
    for r in results[:3]:
        with st.expander(f"#{r['rank']} {r['candidate_id']} — {r['final_score']}/100", expanded=True):
            st.markdown(explainer.explain_candidate(r))

    st.header("⚖️ JD Bias / Narrow-Phrasing Check (bonus feature)")
    flags = st.session_state.get("bias_flags", [])
    if flags:
        for f in flags:
            st.warning(f"**[{f['type']}]** {f['detail']}")
    else:
        st.success("No obvious bias or overly narrow phrasing detected in this JD.")

    st.header("🔎 Ask a recruiter question (bonus feature)")
    st.caption('Type a natural question, e.g. "Why is Candidate_Strong ranked above '
               'Candidate_Medium?" or "Why did Candidate_Weak rank low?"')

    def find_mentioned_candidates(question: str, candidate_ids):
        """Fuzzy-matches candidate ids mentioned anywhere in a free-text question,
        so the recruiter doesn't need to type the exact filename-derived id."""
        q_lower = question.lower()
        found = []
        for cid in candidate_ids:
            cid_norm = re.sub(r"[_\-]", " ", cid).lower()
            cid_tokens = cid_norm.split()
            # match if every token of the candidate id appears in the question,
            # or the raw id string appears directly
            if cid.lower() in q_lower or all(tok in q_lower for tok in cid_tokens):
                found.append(cid)
        return found

    def answer_question(question: str, results):
        ids = [r["candidate_id"] for r in results]
        mentioned = find_mentioned_candidates(question, ids)
        by_id = {r["candidate_id"]: r for r in results}

        if len(mentioned) >= 2:
            x, y = mentioned[0], mentioned[1]
            rx, ry = by_id[x], by_id[y]
            higher, lower = (rx, ry) if rx["final_score"] >= ry["final_score"] else (ry, rx)
            diff_req = set(higher["matched_required"]) - set(lower["matched_required"])
            diff_sem = round(higher["semantic_score"] - lower["semantic_score"], 1)
            lines = [
                f"**{higher['candidate_id']}** (#{higher['rank']}, {higher['final_score']}/100) "
                f"ranks above **{lower['candidate_id']}** (#{lower['rank']}, {lower['final_score']}/100).",
                f"- Required-skill coverage: {higher['required_skill_score']}% vs "
                f"{lower['required_skill_score']}%"
                + (f" — {higher['candidate_id']} additionally matches: {', '.join(diff_req)}"
                   if diff_req else ""),
                f"- Semantic fit to the JD: {higher['semantic_score']}% vs {lower['semantic_score']}% "
                f"({'+' if diff_sem >= 0 else ''}{diff_sem} points)",
            ]
            if lower["missing_required"]:
                lines.append(f"- {lower['candidate_id']} is missing required skills: "
                             f"{', '.join(lower['missing_required'])}")
            return "\n\n".join(lines)

        elif len(mentioned) == 1:
            r = by_id[mentioned[0]]
            return explainer.explain_candidate(r)

        else:
            return ("I couldn't find a matching candidate name in that question. "
                    "Try including a candidate's exact name, e.g. \"" +
                    (ids[0] if ids else "Candidate_1") + "\".")

    user_question = st.text_input("Your question", key="recruiter_question")
    if st.button("Ask") and user_question.strip():
        answer = answer_question(user_question, results)
        st.markdown("---")
        st.markdown(answer)
else:
    st.info("Upload a JD and resumes in the sidebar, then click **Run ranking**.")
