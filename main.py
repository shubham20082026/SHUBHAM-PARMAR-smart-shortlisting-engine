import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src import parser as parser_mod
from src import scorer
from src import explainer
from src import bias_detector


def main():
    ap = argparse.ArgumentParser(description="Rank resumes against a job description.")
    ap.add_argument("--jd", required=True, help="Path to the Job Description PDF")
    ap.add_argument("--resumes", required=True, help="Directory containing resume PDFs")
    ap.add_argument("--out", default="outputs/results.csv", help="Path to write ranked results CSV")
    ap.add_argument("--top", type=int, default=3, help="How many top candidates to explain in detail")
    args = ap.parse_args()

    print(f"Reading JD from {args.jd} ...")
    jd_text = parser_mod.extract_text_from_pdf(args.jd)

    print(f"Loading resumes from {args.resumes} ...")
    resumes = parser_mod.load_resumes(args.resumes)
    print(f"Loaded {len(resumes)} resumes.")

    print("Scoring & ranking (this loads the semantic model on first run)...")
    results = scorer.rank_candidates(jd_text, resumes)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import csv
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "candidate_id", "final_score", "required_skill_score",
                          "preferred_skill_score", "semantic_score",
                          "matched_required", "missing_required"])
        for r in results:
            writer.writerow([
                r["rank"], r["candidate_id"], r["final_score"],
                r["required_skill_score"], r["preferred_skill_score"], r["semantic_score"],
                "; ".join(r["matched_required"]), "; ".join(r["missing_required"]),
            ])
    print(f"\nFull ranking written to {args.out}\n")

    print("=" * 70)
    print("RANKED SHORTLIST")
    print("=" * 70)
    for r in results:
        print(f"#{r['rank']:>2}  {r['candidate_id']:<25} score={r['final_score']:>5}/100  "
              f"required={r['required_skill_score']:>5}%  semantic={r['semantic_score']:>5}%")

    print("\n" + "=" * 70)
    print(f"TOP {args.top} — DETAILED EXPLANATIONS")
    print("=" * 70)
    for text in explainer.explain_top_n(results, n=args.top):
        print("\n" + text)

    print("\n" + "=" * 70)
    print("BONUS: JD BIAS / NARROW-PHRASING CHECK")
    print("=" * 70)
    flags = scorer.analyze_jd_bias(jd_text)
    print(bias_detector.format_bias_report(flags))

    # also dump full JSON for the UI / further inspection
    json_out = os.path.splitext(args.out)[0] + ".json"
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull structured results (for UI/demo) written to {json_out}")


if __name__ == "__main__":
    main()
