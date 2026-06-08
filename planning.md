# Project 1 Planning: The Unofficial Guide

Write this document before you write any pipeline code.
Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation the more specific they are, the more useful the generated code will be.
Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
Update this file before starting any stretch features.

---

## Domain

Northeastern University Khoury College of Computer Sciences graduate course and professor reviews — specifically for core MS CS/AI courses such as CS5800 (Algorithms) and CS5100 (Foundations of AI). This knowledge is valuable because incoming and current students need to make high-stakes registration decisions (which professor to choose, how hard the exams are, how grading works) but this information is scattered across Rate My Professors, Reddit threads, Discord servers, and word-of-mouth. Official course catalogs give zero insight into actual teaching style, exam difficulty, or grade distribution. A student who picks the wrong professor for CS5800 may fail or tank their GPA; there is no single searchable source to prevent that.

---

## Documents

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | Reddit r/NEU | CS5100 / CS5800 professor selection thread (incoming MSAI student asking for recommendations) | reddit_neu_cs5500_thread.txt |
| 2 | Reddit r/NEU | CS5800 Algorithms professor comparison for Spring 2026 (Iraklis vs Mahsa vs Aloupis vs Ravi) | reddit_neu_cs5500_thread.txt |
| 3 | Rate My Professors | Prof. Shesh — CS3100 review (Quality 2.0, Apr 21, 2026) | rmp_smith_cs5800.txt |
| 4 | Rate My Professors | Prof. Shesh — CS3100 review (Quality 4.0, Apr 15, 2026) | rmp_smith_cs5800.txt |
| 5 | Rate My Professors | Prof. Shesh — CS5010 review (Quality 1.0, Dec 19, 2025) | rmp_smith_cs5800.txt |
| 6 | Rate My Professors | Prof. Shesh — CS3500 review (Quality 4.0, Jun 17, 2025) | rmp_smith_cs5800.txt |
| 7 | Rate My Professors | Prof. Shesh — CS3500 review (Quality 2.0, Jun 16, 2025) | rmp_smith_cs5800.txt |
| 8 | Rate My Professors | Prof. Shesh — CS5010 review (Quality 4.0, Dec 19, 2024) | rmp_smith_cs5800.txt |
| 9 | Rate My Professors | Prof. Shesh — CS5010 review (Quality 1.0, Nov 2024) | rmp_smith_cs5800.txt |
| 10 | Rate My Professors | Prof. Shesh — CS3500 review (Quality 5.0, Jul 8, 2024) | rmp_smith_cs5800.txt |
| 11 | Rate My Professors | Prof. Shesh — CS3500 review (Quality 5.0, Jun 30, 2024) | rmp_smith_cs5800.txt |

---

## Chunking Strategy

Chunk size: 300 characters (approximately 50–70 tokens)
Overlap: 50 characters
Reasoning:
My corpus contains two structurally different document types, which informs this decision:

RMP reviews (short, opinion-dense): Each review is 2–5 sentences and focuses on a single student's experience. A 300-character chunk captures roughly one full review or one coherent opinion unit. Chunking smaller (e.g. 100 chars) would split a review mid-sentence, losing the evaluative conclusion. Chunking larger would merge multiple reviews from different students into one chunk, making it impossible to attribute a specific claim to a specific source.

Reddit threads (medium-length comments): Individual comments range from 2 to 8 sentences. A 300-character chunk captures one commenter's key point. The 50-character overlap ensures that if a sentence spans a chunk boundary (e.g. "Take Mahsa if you want an easier A — / — but Iraklis if you want to learn deeply"), both adjacent chunks contain enough context for a meaningful embedding.

This chunk size is intentionally on the smaller end because the key facts in this domain are concentrated (a rating, a specific claim about exams, a grading policy) rather than spread across paragraphs. Larger chunks would dilute the signal.

---

## Retrieval Approach

**Embedding model:**

Embedding model: all-MiniLM-L6-v2 via sentence-transformers (runs locally, no API key)
Top-k: 5
Reasoning for top-k: Student queries often ask about a professor across multiple courses or from multiple reviewer perspectives. Retrieving 5 chunks allows the LLM to synthesize 2–3 relevant opinions without over-diluting the context with loosely related material. If initial testing shows the top results are redundant (e.g. 5 chunks all from the same review file), I will reduce to k=3 and rely on metadata filtering by professor name instead.
Production tradeoff reflection:
If deploying this for real NEU students at scale, I would consider the following tradeoffs when choosing an embedding model:

text-embedding-3-small (OpenAI API): Higher accuracy on domain-specific jargon and short opinionated text than MiniLM, but costs money per query and requires an API key — unsuitable for a fully local prototype.
all-mpnet-base-v2: Stronger semantic accuracy than MiniLM, still local, but ~3× slower on CPU — acceptable for a low-traffic app, problematic at scale.
Context length: MiniLM handles 256 tokens; for longer Reddit threads (500+ tokens), I would need a model with a larger context window (e.g. e5-large) to avoid truncation artifacts.
Multilingual support: NEU's student body includes many international students who may write reviews in Chinese or other languages. MiniLM is English-only; paraphrase-multilingual-MiniLM-L12-v2 would be a drop-in replacement with multilingual support if needed.

---

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What do students say about Prof. Iraklis Tsekourakis's exams in CS5800? | Mixed: some say exams are very hard (100 pts / 4 problems, failing likely), others say they are easy if you study slides and do homework. A TA confirmed 3 exams based entirely on class slides. |
| 2 | How does Prof. Mahsa Derakhshan grade exams in CS5800? | Exams are fair; she curved the first exam; exam questions resemble practice problems she provides; no trick questions; focuses on understanding of key concepts. |
| 3 | What are the main criticisms of Prof. Shesh in CS5010? | He does not respond to student emails; grading is extremely strict with minimal room for error; TAs are not well supervised; the grading scheme does not match the stated learning goals. |
| 4 | If I want a better GPA in CS5800, which professor should I choose? | Students suggest Prof. Mahsa Derakhshan for a more manageable path to an A; Prof. Iraklis for deeper learning but more exams. Prof. Aloupis is warned against for GPA-conscious students due to very hard tests and heavy workload. |
| 5 | What is the workload like for Prof. Shesh's CS3500 (OOD) course? | Very high homework load — one reviewer reports 20-40 hours per week after assignment 3. Lectures are highly rated. Grading is strict but regrade requests are respected. |

---

## Anticipated Challenges

1. Chunk boundary splits key facts: Several Reddit comments contain a compound claim across two sentences (e.g. "Take Mahsa if you want an easier A — take Iraklis if you want to learn deeply"). If this spans a chunk boundary, each half loses context. The 50-character overlap is designed to mitigate this, but I will manually inspect boundary chunks during Milestone 3 to verify.

2. Multiple professors, same course — retrieval conflation: Queries about CS5800 may retrieve chunks about Iraklis, Mahsa, Aloupis, and Ravi simultaneously, and the LLM may blend their attributes into a single confused answer. I will include professor name as chunk metadata and test whether adding "Professor Mahsa" to a query improves precision. If conflation persists, a metadata filter by professor name is a stretch feature worth implementing.

3. Sparse coverage for some professors: I have strong review data for Prof. Shesh and partial data for CS5800 professors, but limited data for CS5100 professors (Prof. Yu, Prof. Tahmasebi). Queries about CS5100 may produce low-confidence retrievals or trigger the "I don't have enough information" fallback — which is actually the correct behavior, but needs to be tested explicitly.

4. Noisy Reddit formatting: Reddit comments include quoted text ("> Prior comment"), markdown syntax, and conversational filler ("lol", "tbh") that adds noise to embeddings without semantic value. Cleaning must strip these before chunking.

---

## Architecture

┌─────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                             │
└─────────────────────────────────────────────────────────────────┘

 [1] Document Ingestion          [2] Chunking
 ┌──────────────────┐            ┌──────────────────────────┐
 │  .txt files      │            │  chunk_size = 300 chars  │
 │  (RMP reviews,   │──────────▶ │  overlap    = 50 chars   │
 │   Reddit threads)│            │  lib: custom Python or   │
 │  Tool: open()    │            │  LangChain CharSplitter  │
 └──────────────────┘            └────────────┬─────────────┘
                                              │
                 ┌────────────────────────────▼──────────────────┐
                 │  [3] Embedding + Vector Store                  │
                 │  Embed: sentence-transformers/all-MiniLM-L6-v2 │
                 │  Store: ChromaDB (local)                       │
                 │  Metadata per chunk: source file, professor,   │
                 │                      course, chunk index       │
                 └────────────────────────┬──────────────────────┘
                                          │
                          ┌───────────────▼──────────────┐
                          │  [4] Retrieval               │
                          │  Query → embed query         │
                          │  → cosine similarity search  │
                          │  → return top-k=5 chunks     │
                          │  + source metadata           │
                          └───────────────┬──────────────┘
                                          │
                          ┌───────────────▼──────────────┐
                          │  [5] Generation              │
                          │  LLM: Groq llama-3.3-70b     │
                          │  Prompt: answer ONLY from    │
                          │  retrieved context; cite     │
                          │  source document in response │
                          │  Interface: Gradio web UI    │
                          └──────────────────────────────┘

---

## AI Tool Plan

**Milestone 3 — Ingestion and chunking:**

Tool: Claude
Input: This planning.md (Documents section + Chunking Strategy section) + one sample .txt file
Ask Claude to implement ingest.py: a script that loads all .txt files from a /data directory, strips Reddit markdown (blockquotes >, *bold*, --- separators) and RMP boilerplate (tag lines, "For Credit: Yes" metadata lines), then chunks the cleaned text using the specified 300-char / 50-char overlap parameters, attaches metadata (source filename, professor name if detectable, chunk index), and writes chunks to a JSON file.
Verification: Manually print 5 random chunks and confirm each is readable, self-contained, and free of formatting artifacts. Check total chunk count is between 20 and 2000.

Note on chunk count range (revised during Milestone 3): Initially I set the lower bound at 50, anticipating a wider corpus. In practice my source material is two consolidated .txt files (~8KB total) representing 11 distinct review/comment entries per the Documents table above — which satisfies the rubric's "≥10 documents" requirement at the entry level even though the file count is 2. With chunk_size=300 and overlap=50, this yielded 27 chunks. I considered shrinking chunk_size to ~150 to mechanically hit the original 50 threshold, but rejected that because shorter chunks would dilute the per-chunk context that the 300-char size was deliberately chosen to preserve (one complete RMP review or one Reddit comment per chunk). Lowering the verification floor to 20 reflects the actual corpus size without sacrificing chunk semantics.

**Milestone 4 — Embedding and retrieval:**

Tool: Claude
Input: This planning.md (Retrieval Approach section + Architecture diagram) + the JSON output from Milestone 3
Ask Claude to implement embed.py: load chunks from JSON, embed with SentenceTransformer("all-MiniLM-L6-v2"), store in ChromaDB collection named neu_reviews with metadata fields source, professor, course, chunk_index. Then implement retrieve.py: a function retrieve(query: str, k: int = 5) -> list[dict] that embeds the query, queries ChromaDB, and returns top-k chunks with text and metadata.
Verification: Run 3 of my 5 evaluation questions manually, print retrieved chunks, confirm they are topically relevant and distance scores are below 0.5.

**Milestone 5 — Generation and interface:**

Tool: Claude
Input: This planning.md (grounding requirement, output format) + retrieve.py interface + Gradio skeleton
Ask Claude to implement app.py: a Gradio interface with a question textbox and "Ask" button, calling retrieve() then passing chunks as context to Groq llama-3.3-70b-versatile with a system prompt that enforces grounding ("Answer only from the provided documents. If the documents do not contain enough information, say so explicitly. Cite the source document name in your answer."). Output fields: Answer (multiline textbox) + Sources (textbox listing retrieved document names).
Verification: Test the "out-of-scope" case (ask about a professor not in my documents) and confirm the system refuses rather than hallucinating.
