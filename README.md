# The Unofficial Guide — Project 1

A RAG system that makes student-generated knowledge about Northeastern University Khoury College graduate CS courses (CS5800, CS5100, CS5010, CS3500, CS3100) searchable. Asks plain-language questions about professors and exams; returns grounded, cited answers drawn from collected Reddit and RateMyProfessors reviews.

**Stack:** sentence-transformers `all-MiniLM-L6-v2` (embeddings) · ChromaDB (vector store) · Groq `llama-3.3-70b-versatile` (generation) · Gradio (UI).

**Run locally:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then paste your Groq key
python ingest.py       # documents/*.txt → chunks.json
python embed.py        # chunks.json → chroma_db/
python app.py          # Gradio UI at http://127.0.0.1:7860
```

---

## Domain

Northeastern University Khoury College of Computer Sciences graduate course and professor reviews — specifically for core MS CS/AI courses such as CS5800 (Algorithms), CS5100 (Foundations of AI), and CS5010 (Programming Design Paradigm).

This knowledge is valuable because incoming and current students need to make high-stakes registration decisions (which professor to choose, how hard the exams are, how grading works), but the information is scattered across Rate My Professors, Reddit threads, Discord servers, and word-of-mouth. Official course catalogs reveal nothing about teaching style, exam difficulty, or grade distribution. A student who picks the wrong professor for CS5800 may struggle or tank their GPA; there is no single searchable source to prevent that.

---

## Document Sources

Two consolidated `.txt` files containing 11 distinct review/comment entries from two sources:

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Reddit r/NEU — CS5100/CS5800 professor selection thread (incoming MSAI) | Forum thread | `documents/reddit_neu_cs5500_thread.txt` |
| 2 | Reddit r/NEU — CS5800 Algorithms prof comparison for Spring 2026 (Iraklis vs Mahsa vs Aloupis vs Ravi) | Forum thread | `documents/reddit_neu_cs5500_thread.txt` |
| 3 | Rate My Professors — Prof. Shesh, CS3100 (Q=2.0, Apr 21 2026) | Review | `documents/rmp_smith_cs5800.txt` |
| 4 | Rate My Professors — Prof. Shesh, CS3100 (Q=4.0, Apr 15 2026) | Review | `documents/rmp_smith_cs5800.txt` |
| 5 | Rate My Professors — Prof. Shesh, CS5010 (Q=1.0, Dec 19 2025) | Review | `documents/rmp_smith_cs5800.txt` |
| 6 | Rate My Professors — Prof. Shesh, CS3500 (Q=4.0, Jun 17 2025) | Review | `documents/rmp_smith_cs5800.txt` |
| 7 | Rate My Professors — Prof. Shesh, CS3500 (Q=2.0, Jun 16 2025) | Review | `documents/rmp_smith_cs5800.txt` |
| 8 | Rate My Professors — Prof. Shesh, CS5010 (Q=4.0, Dec 19 2024) | Review | `documents/rmp_smith_cs5800.txt` |
| 9 | Rate My Professors — Prof. Shesh, CS5010 (Q=1.0, Nov 2024) | Review | `documents/rmp_smith_cs5800.txt` |
| 10 | Rate My Professors — Prof. Shesh, CS3500 (Q=5.0, Jul 8 2024) | Review | `documents/rmp_smith_cs5800.txt` |
| 11 | Rate My Professors — Prof. Shesh, CS3500 (Q=5.0, Jun 30 2024) | Review | `documents/rmp_smith_cs5800.txt` |

---

## Chunking Strategy

**Chunk size:** 300 characters (~50–70 tokens)

**Overlap:** 50 characters

**Preprocessing before chunking** (see `ingest.py`):
- Strip UTF-8 BOM if present
- Drop RMP boilerplate lines: `For Credit:`, `Attendance:`, `Would Take Again:`, `Grade:`, `Textbook:`, `Tags:`, `Quality:`, `Difficulty:`, `N helpful`/`thumbs up`
- Strip Reddit blockquote markers (`>`), horizontal rules (`---`, `***`), `**bold**`, `*italic*`, markdown links `[text](url)`, and bare URLs
- Collapse runs of whitespace and blank lines
- Trim each line

**Why these choices fit your documents:**

The corpus has two structurally different document types, and 300 chars works for both:

- **RMP reviews** are 2–5 sentences focused on a single student's experience. A 300-char chunk captures roughly one full review or one coherent opinion unit. Going smaller (e.g., 100 chars) would split a review mid-sentence and lose the evaluative conclusion; going larger would merge multiple students' reviews into one chunk, making it impossible to attribute a specific claim to a specific source.
- **Reddit comments** range from 2–8 sentences. The 50-char overlap is designed to bridge sentences that span a chunk boundary — the most important example in this corpus is the compound claim *"Take Mahsa if you want an easier A — take Iraklis if you want to learn deeply"*, where each half loses meaning without the other.

Metadata attached per chunk: `source` (filename), `professor` (matched against a known-names list), `course` (regex `\bCS\s?\d{4}\b`), and `chunk_index`.

**Final chunk count:** **27 chunks** (Reddit: 14, RMP: 13). This is below the original 50-chunk floor in `planning.md` because the source files total only ~8 KB. The floor was revised to 20 in `planning.md` rather than shrinking `chunk_size` to mechanically hit 50, because shorter chunks would dilute the per-chunk context that 300 chars was deliberately chosen to preserve.

---

## Embedding Model

**Model used:** `sentence-transformers/all-MiniLM-L6-v2` — 384-dim embeddings, runs locally on CPU in seconds for 27 chunks, no API key, no rate limits.

**Production tradeoff reflection:**

If deploying this for real NEU students at scale, I'd weigh several alternatives against MiniLM:

- **`text-embedding-3-small` (OpenAI):** Higher accuracy on domain jargon and short opinionated text, but costs money per query and requires a managed API key — unsuitable for a fully local prototype and probably overkill for a corpus this small.
- **`all-mpnet-base-v2`:** Stronger semantic accuracy than MiniLM, still local, but ~3× slower on CPU. Acceptable for a low-traffic app, problematic if rebuilding embeddings on every doc update at scale.
- **Context length:** MiniLM truncates at 256 tokens. Our chunks are well under that limit (300 chars ≈ 50–70 tokens), so this is not currently a constraint — but if a future iteration kept whole Reddit threads as single chunks (500+ tokens), `e5-large` or `bge-large-en-v1.5` would be needed.
- **Multilingual support:** NEU's student body includes many international students who may write reviews in Chinese or other languages. MiniLM is English-only; `paraphrase-multilingual-MiniLM-L12-v2` is a near drop-in replacement if non-English content becomes a meaningful share of the corpus.

For the current 27-chunk prototype, MiniLM's speed and zero-cost outweigh the marginal accuracy of larger models. The decision would flip if we expanded to thousands of reviews across many universities.

---

## Grounded Generation

**System prompt grounding instruction** (verbatim from `app.py`):

```
You are an assistant that answers questions about Northeastern University Khoury College
graduate courses and professors, using ONLY the document excerpts provided in the user message.

Rules — follow them strictly:
1. Answer only from the provided excerpts. Do not use outside knowledge, even if you "know"
   the answer.
2. If the excerpts do not contain enough information to answer, reply exactly:
   "I don't have enough information in the provided documents to answer that."
   Do not guess, infer beyond what is written, or fabricate professor names, courses, or reviews.
3. Cite the source document filename(s) inline in parentheses next to each claim,
   e.g. "(rmp_smith_cs5800.txt)". If multiple excerpts support the same claim, cite all of them.
4. When excerpts conflict (e.g., one says exams are easy, another says hard), present
   both perspectives rather than picking one.
5. Keep the answer concise — 2–5 sentences unless the question explicitly asks for more.
```

**Structural choices that reinforce grounding:**

- The user message is formatted as `Question: ... \n\nDocument excerpts: [Excerpt 1] source=... | professor=... | course=... \n {text} ...` so the model sees per-chunk metadata it can cite from explicitly.
- `temperature=0.1` minimizes creative drift away from the retrieved context.
- A literal refusal sentence is specified so failure cases produce a consistent, machine-detectable string ("I don't have enough information...") rather than varied hedges.

**How source attribution is surfaced in the response:**

Two channels:
1. **Inline citations** inside the Answer text — e.g., `"...grading is extremely strict (rmp_smith_cs5800.txt)"`, produced by Rule 3 of the system prompt.
2. **Sources panel** in the Gradio UI — a deduplicated list of every filename returned by the retriever (in retrieval order), rendered separately from the Answer textbox.

---

## Evaluation Report

All 5 evaluation questions were run end-to-end against the deployed system. Retrieval used `k=5` cosine similarity; generation used `llama-3.3-70b-versatile` at `temperature=0.1`.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What do students say about Prof. Iraklis Tsekourakis's exams in CS5800? | Mixed: some say very hard (100 pts / 4 problems), others easy if you study slides; a TA confirmed 3 exams based on slides | **Refused** ("not enough information") despite retrieving chunks 8/10 of the Reddit thread that contain exactly this info | Relevant (top-5 distances 0.37–0.48; the substantive chunks were retrieved) | **Inaccurate** — refused when chunks supported an answer; see Failure Case below |
| 2 | How does Prof. Mahsa Derakhshan grade exams in CS5800? | Fair, curved first exam, exams resemble practice problems, no trick questions | **Refused** despite the top-1 chunk (d=0.33) containing the verbatim answer | Relevant (top-1 was the right chunk) | **Inaccurate** — same over-refusal pattern as Q1 |
| 3 | What are the main criticisms of Prof. Shesh in CS5010? | Doesn't reply to emails; extremely strict grading; minimal room for error; TAs under-supervised | Captured "doesn't respond to emails" + "extremely strict" + "unpolished syllabus", with a thoughtful caveat that other CS3500 reviews of Shesh are positive | Relevant (all 5 top chunks are Shesh reviews; top-1 is the CS5010 negative review) | **Accurate** |
| 4 | If I want a better GPA in CS5800, which professor should I choose? | Mahsa for easier A; Iraklis for deeper learning; Aloupis warned against (very hard tests, heavy workload) | Recommended **Iraklis** for GPA — the opposite of the expected answer. The key "Take Mahsa if you want an easier A" chunk (idx=7) was not in top-5 | Partially relevant (top-5 distances 0.44–0.59 — the worst of any question; Yu in CS5100 retrieved at rank 1) | **Inaccurate** — retrieval miss propagated to a wrong recommendation |
| 5 | What is the workload like for Prof. Shesh's CS3500 (OOD) course? | Very high — 20–40 hours/week after assignment 3; great lectures; grading strict but regrades respected | Captured "hardest course in schedule" warning, "PREPARE YOURSELF..." quote, and the late-days/regrades concession | Relevant (all 5 chunks are Shesh; 3/5 are CS3500) | **Accurate** |

**Summary:** 2/5 accurate (Q3, Q5), 1/5 partially accurate retrieval but wrong recommendation (Q4), 2/5 over-refusal (Q1, Q2). The accurate cases are the ones where the right chunk both retrieves at top-1 *and* names the course explicitly in the chunk body.

---

## Failure Case Analysis

**Question that failed:** Q1 — "What do students say about Prof. Iraklis Tsekourakis's exams in CS5800?"

**What the system returned:**

> "I don't have enough information in the provided documents to answer that, as the excerpts provided are about CS5100 or CS5500, but not CS5800... The only mention of CS5800 is in a filename (`rmp_smith_cs5800.txt`), but it does not contain information about Prof. Iraklis Tsekourakis's exams."

**Root cause (tied to a specific pipeline stage):**

This is a *chunking / metadata* failure that *cascades into a generation* failure.

The Reddit source document opens with a header block: `"CS5800 (Algorithms) professor comparison for Spring 2026... Options: Iraklis Tsekourakis, Aloupis, Mahsa Derakhshan, Ravi Sundaram."`. The substantive content about Iraklis's exams ("100 points across only 4 problems", "3 exams entirely based on slides", "many students failed") sits later in the document, in chunks 8 and 10. By the time the chunker reaches those chunks, the 300-char window no longer overlaps with the "CS5800" header — so the chunk text doesn't contain the string "CS5800".

Compounding this, my `detect_course()` function in `ingest.py` takes the *first* `CS\d{4}` match it finds within a chunk. Chunks 8 and 10 either contain no course code (course metadata = `''`) or contain `CS5100` from an unrelated nearby sentence — never `CS5800`.

The retriever still ranks these chunks correctly by semantic similarity (distances 0.44 and 0.47, comfortably below the 0.5 threshold). But the generation system prompt is strict — "Do not guess, infer beyond what is written, or fabricate" — and the model interprets the absence of "CS5800" in either the chunk body or the metadata as evidence that the excerpts are off-topic. So it correctly applies Rule 2 and refuses, even though a human reading the same excerpts would clearly attribute them to Iraklis's CS5800 exams.

**What you would change to fix it:**

Two complementary fixes, in order of impact:

1. **Carry document-level context into every chunk** — prepend a short header (the source's topic line, e.g., `"[Context: CS5800 Algorithms professor comparison]"`) to every chunk derived from that document. This restores the course anchor that gets lost as chunks move away from the document header. Implementation: a one-line per-document prefix in `ingest_file()`.
2. **Improve `detect_course()`** — when a chunk has no `CS\d{4}` match, fall back to the most-recent course code seen in the same document above this chunk (a stateful pass during `ingest_file`). This would have tagged chunks 8 and 10 as `CS5800` correctly.

I deliberately did *not* weaken the grounding rules in the system prompt to fix this — the refusal behavior is correct given the inputs, and softening Rule 2 would risk hallucinations on genuinely out-of-scope queries (which the system currently handles correctly, as verified in the out-of-scope smoke test with "Prof. Beyonce Knowles in CS9999").

---

## Spec Reflection

**One way the spec helped you during implementation:**

The Chunking Strategy section of `planning.md` forced me to articulate *why* 300/50 fit this specific corpus before I wrote any code — namely that RMP reviews are short and opinion-dense, so each chunk should hold one coherent review. That framing translated directly into the cleaning regexes in `ingest.py`: I knew to strip `For Credit:`, `Quality:`, etc. because the spec said the chunk should be "one full review", and those tag lines would otherwise dominate the embedding. Without the spec, I'd likely have shipped a generic character splitter and discovered the boilerplate problem only after retrieval failed.

**One way your implementation diverged from the spec, and why:**

The spec set a verification floor of "50–2000 chunks", but the real corpus produces 27. Rather than shrink `chunk_size` to mechanically clear 50, I updated `planning.md` to lower the floor to 20 and documented the rationale (the source files are 11 review entries totaling 8 KB; smaller chunks would dilute the per-chunk context that 300 chars was deliberately chosen to preserve). The spec's verification target was a reasonable initial assumption that didn't survive contact with the actual corpus size — and changing the spec, not the implementation, was the right correction.

---

## AI Usage

**Instance 1 — Ingestion + chunking (`ingest.py`)**

- *What I gave the AI:* The Documents section and Chunking Strategy section from `planning.md`, plus the constraint that the script must produce `chunks.json` with per-chunk metadata `{source, professor, course, chunk_index}`.
- *What it produced:* A Python script with `clean_text()` (regex stripping for Reddit and RMP noise), `chunk_text()` (fixed-size character chunking with overlap), `detect_professor()`/`detect_course()` for metadata, and a `main()` that wrote JSON and printed 5 random chunks for verification.
- *What I changed or overrode:* Two things. (1) The initial RMP boilerplate regex missed a UTF-8 BOM character that showed up at the start of `chunks.json` chunk 0 — I added an explicit `raw.lstrip("﻿")` rather than expanding the regex, since the BOM is a file-level artifact, not a content pattern. (2) When the verification step warned that 27 chunks fell below the planned 50-floor, the AI suggested either shrinking `chunk_size` or adding more documents. I rejected both — shrinking would have broken the "one review per chunk" property, and adding more docs was out of scope — and instead updated `planning.md` to lower the floor with a written rationale. This was a spec-correction, not a code change.

**Instance 2 — Grounded generation (`app.py`)**

- *What I gave the AI:* The grounding requirement from `planning.md` ("answer only from retrieved context; cite source document"), the `retrieve(query, k=5)` signature from Milestone 4, and a list of the 5 evaluation questions.
- *What it produced:* A Gradio app with a 3-rule grounding system prompt and a single `answer` textbox combining response + sources.
- *What I changed or overrode:* I rewrote the system prompt from 3 rules to 5 — adding (a) a literal refusal sentence template ("I don't have enough information in the provided documents to answer that.") so failure cases are machine-detectable, and (b) explicit handling for conflicting excerpts ("present both perspectives rather than picking one"), because the corpus has known disagreements (e.g., Iraklis exams "very hard" vs "easy if you study slides"). I also split the output into two distinct fields (Answer + Sources) rather than concatenating them, because the rubric requires source attribution to be visible independently of the prose. When evaluation surfaced over-refusal on Q1/Q2, I resisted the temptation to soften Rule 2 — instead documenting the behavior as the primary failure case, since loosening the rule would risk hallucinations on out-of-scope queries.
