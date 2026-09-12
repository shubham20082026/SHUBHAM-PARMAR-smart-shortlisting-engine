# (nexorahackathonmitblr)Smart Shortlisting Engine

Hybrid semantic + keyword resume ranker built for the InternLoom AI Hackathon
(Manipal Institute of Technology).

## Quick start

```bash
pip install -r requirements.txt

# Put the sample JD + resumes here (or point --jd / --resumes elsewhere):
#   data/Sample_JD.pdf
#   data/resumes/*.pdf

python main.py --jd data/Sample_JD.pdf --resumes data/resumes --out outputs/results.csv
```

This prints the full ranking + top-3 explanations to the terminal, and writes
`outputs/results.csv` and `outputs/results.json`.

For the live demo:

```bash
streamlit run app.py
```

Upload the JD and resumes in the sidebar and click **Run ranking**. Includes
the bonus "why is X ranked above Y" comparison tool.

> **Note on the semantic model:** on first run, `sentence-transformers` downloads
> `all-MiniLM-L6-v2` (~90MB) — make sure you do this once *before* you're
> offline/on stage. If it can't load (no internet, not installed), the system
> **automatically falls back to TF-IDF cosine similarity** so the demo never
> breaks — you'll see a `[semantic] falling back to TF-IDF` message in that
> case. Test this ahead of time and know which mode you're demoing in.

---

## How the matching actually works 

### 1. Parsing (`src/parser.py`, `src/jd_parser.py`)
- Resume and JD PDFs are converted to plain text with `pdfplumber`.
- The JD is split into a `required` / `preferred` / `general` section using
  header keyword detection (e.g. "Requirements", "Nice to have").
- The JD is also broken into individual **requirement chunks** (one per
  bullet/sentence) — this granularity matters for semantic matching (see
  below): comparing one JD line at a time against a resume is far more
  precise than comparing two giant blobs of text.

### 2. Keyword arm (`src/skills.py`) — 45% required + 15% preferred = 60% of score
- We extract explicit skills from the JD by scanning it against a curated
  ~90-term technical vocabulary (languages, frameworks, DBs, tools, cloud,
  etc.), covering common aliasing (e.g. `React.js` / `ReactJS` / `React` all
  canonicalize to `react`).
- For each resume, every required/preferred skill is checked for presence
  using **normalized + fuzzy string matching** (`difflib.SequenceMatcher`),
  so "Expres.js" (typo) or "ExpressJS" (no punctuation) still count as a hit
  for `express`.
- `keyword_score = (# required skills matched) / (# required skills in JD)`,
  computed separately for required vs. preferred skills.

### 3. Semantic arm (`src/semantic.py`) — 40% of score
- Both the JD requirement chunks and the resume (chunked into
  bullets/sentences via `parser.chunk_text`) are embedded with
  `sentence-transformers` (`all-MiniLM-L6-v2`).
- We compute cosine similarity between **every JD requirement chunk** and
  **every resume chunk**, take the *best-matching resume chunk* for each JD
  requirement, then average those best-matches across all requirements.
- This is the design decision that solves the problem statement's core
  example: a resume saying *"built REST APIs with Express and MongoDB"* will
  score highly against a JD line like *"Node.js backend development"* purely
  from embedding similarity — no shared keywords needed.
- Why "best match per requirement" instead of one whole-document embedding?
  Because a resume also contains irrelevant content (education, hobbies,
  unrelated jobs) that would dilute a single averaged embedding. Taking the
  best evidence per requirement avoids that dilution and also gives us a
  ready-made citation for the explanation step.

### 4. Combining the two (`src/scorer.py`)
```
final_score = 0.45 × required_keyword_score
            + 0.15 × preferred_keyword_score
            + 0.40 × semantic_score
```
- Required-keyword coverage gets the single largest weight because the
  problem statement explicitly says a role that names specific required
  skills "shouldn't be satisfied by only loosely related experience."
- Semantic score is weighted almost as heavily so genuinely relevant
  experience phrased differently from the JD isn't penalized.
- Preferred/nice-to-have skills matter least, since they're explicitly
  optional in the JD.
- All three weights are named constants at the top of `scorer.py` — change
  them in one place, and be ready to justify the numbers (we did, above).

### 5. Explanations (`src/explainer.py`)
- Fully template-based, not LLM-generated prose — every line traces back to
  a number the system computed: matched/missing required skills, matched
  bonus skills, the overall semantic %, and the single strongest semantic
  evidence pair (JD line ↔ resume line + similarity score) so a recruiter
  can see *exactly* why that line pulled the score up.

### Bonus features implemented

**1. JD bias / narrow-phrasing flagging** (`src/bias_detector.py`)
Rule-based, explainable checks run once per JD:
- Gendered/exclusionary language (e.g. "ninja", "rockstar") — flagged per known
  HR research on job-ad language that discourages some applicants.
- Overly narrow tool requirements named without an "or equivalent" hedge (e.g.
  requiring "React" by name when the real need is "modern frontend framework
  experience") — maps each required skill to its general competency category.
- Seniority mismatches: an experience-year floor (e.g. "5+ years") or a hard
  "Bachelor's required" line on a posting whose title says intern/junior/entry-level.
- Excessive simultaneous required-tool lists (8+ named technologies) that risk
  filtering out strong generalists.
Shown in the terminal output (`main.py`) and as an expandable warning panel in
the Streamlit app.

**2. Recruiter Q&A chat** (`app.py`)
A free-text box (not just a dropdown) — fuzzy-matches candidate names mentioned
anywhere in a typed question (handles underscores/spacing variants), then:
- Two candidates mentioned → generates a grounded "X ranks above Y because..."
  comparison citing the actual skill/semantic score deltas.
- One candidate mentioned → returns that candidate's full top-3-style explanation.
- No match → asks for a valid candidate name rather than guessing.
All answers are template-based from the real score breakdown, same
grounded-in-data philosophy as the top-3 explanations.

**3. Messy resume formatting robustness** (`src/parser.py`)
- **OCR fallback**: if `pdfplumber` extracts almost no text (a strong signal
  the PDF is a scanned image rather than real text), the pipeline automatically
  retries with `pytesseract` + `pdf2image` OCR rather than silently scoring an
  empty resume as a zero match. (Requires `pytesseract` and `poppler` installed
  on the machine — falls back to a no-op that keeps whatever text was found if
  OCR deps aren't available, so it never crashes the pipeline.)
- **Section header normalization**: a lookup table collapses inconsistent
  headers ("Tech Stack" / "Toolbox" / "Core Competencies" → `skills`, etc.)
  so downstream logic isn't confused by phrasing differences across resumes.
- **Flexible date-range detection**: a single regex handles "Jan 2022 -
  Present", "01/2022 - 03/2023", and "2022-2023" style variants uniformly.
- **Typos/variant tool names**: already handled by the fuzzy skill matcher in
  `skills.py` (`difflib.SequenceMatcher`), so "Expres.js" or "ExpressJS" still
  register as `express`.
- **Graceful fallback**: if the semantic model can't load, TF-IDF keeps the
  whole pipeline functional rather than crashing.

---

## Project structure
```
smart_shortlist/
├── main.py              # CLI: JD + resume folder -> ranked CSV/JSON + explanations
├── app.py                # Streamlit demo UI (bonus: recruiter Q&A)
├── requirements.txt
├── data/
│   ├── Sample_JD.pdf     # <- put your JD here
│   └── resumes/*.pdf     # <- put your resumes here
├── outputs/               # results.csv / results.json land here
└── src/
    ├── parser.py          # PDF -> text, text chunking
    ├── jd_parser.py        # JD section + requirement-chunk splitting
    ├── skills.py           # skill vocabulary, extraction, fuzzy matching
    ├── semantic.py          # embeddings / TF-IDF fallback, cosine similarity
    ├── scorer.py            # combines keyword + semantic into final ranking
    └── explainer.py          # top-N explanation generation
```

## Rubric alignment
| Criterion | Weight | Where it's addressed |
|---|---|---|
| Semantic + keyword matching | 35% | `skills.py` (keyword) + `semantic.py` (embeddings), combined in `scorer.py` |
| Quality/sensibility of ranking | 20% | Weighted combination tuned to problem statement's own stated priorities |
| Top-3 explanation accuracy/clarity | 20% | `explainer.py`, grounded in real matched/missing data + cited evidence |
| Working end-to-end demo | 15% | `main.py` CLI + `app.py` Streamlit UI |
| Bonus | 10% | JD bias/narrow-phrasing detector (`bias_detector.py`), free-text recruiter Q&A chat, OCR fallback for scanned resumes, section-header normalization |
